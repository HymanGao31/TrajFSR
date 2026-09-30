import torch
import torch.nn as nn
import numpy as np

from models.layers import MLP, st_encoder_with_attn
from models.fsr_traj import TrajFSR

class MRF_CVAE(nn.Module):
    def __init__(self, fut_step: int=20, z_dim: int=16, f2_dim: int=32, sigma1: float=1., sigma2: float=1., stride: int=5, N: int=20, use_fsr: bool=False, fsr_tau: float=0.1):
        super(MRF_CVAE, self).__init__()

        self.z_dim = z_dim
        self.hist_dim = 256

        self.encoder_past = st_encoder_with_attn(n_layer=1)   # hidden_dim=256
        self.update_decoder = MLP(in_feat=256+z_dim, out_feat=2*stride, hid_feat=(512, 512, 256), activation=nn.ReLU())

        self.f2_dim = f2_dim  # feature dim for whole-space configuration
        self.encoder_space_state = MLP(in_feat=2*stride, out_feat=f2_dim, hid_feat=(32, 32), activation=nn.ReLU())
        self.dynamics_decoder = MLP(in_feat=f2_dim+z_dim, out_feat=2*stride, hid_feat=(128, 128, 64), activation=nn.ReLU())

        self.edge_potential_net = MLP(in_feat=3, out_feat=f2_dim, hid_feat=(16, 32), activation=nn.ReLU())

        # for training
        self.encoder_future_eachstep = MLP(in_feat=2*stride, out_feat=2*z_dim, hid_feat=(64, 256, 256), activation=nn.ReLU())

        # for sampling
        self.sigma1 = sigma1
        self.sigma2 = sigma2

        self.sample_past = MLP(in_feat=256, out_feat=z_dim*N, hid_feat=(128, 512), activation=nn.ReLU())
        self.sample_spacestate = MLP(in_feat=f2_dim, out_feat=z_dim, hid_feat=(32, 64), activation=nn.ReLU()) # input tensor is extended with N

        self.fut_step = fut_step
        self.stride = stride

        self.use_fsr = bool(use_fsr)
        self.fsr = TrajFSR(self.hist_dim, hidden=self.hist_dim, tau=float(fsr_tau)) if self.use_fsr else None
        self._fsr_aux = None

    def encode_history(self, proc_data_dict):
        """History Encoder output [P, 256], before TrajFSR."""
        Zt = proc_data_dict['past_traj']
        return self.encoder_past(Zt, proc_data_dict['traj_mask'])

    def _apply_fsr(self, history_feature):
        self._fsr_aux = None
        if self.fsr is None:
            return history_feature
        feat = history_feature.unsqueeze(1)  # [P, 1, 256]
        r_feat, nr_feat, rec_feat, mask, rec_units = self.fsr(feat, is_eval=not self.training)
        self._fsr_aux = {
            "enc": feat,
            "f+": r_feat,
            "f-": nr_feat,
            "rec": rec_feat,
            "mask": mask,
            "rec_units": rec_units,
        }
        return (r_feat + rec_feat).squeeze(1)

    def forward(self, proc_data_dict, N, vae_train=True, noise=False,
                history_feature_override=None, z1_override=None):
        '''
        proc_data_dict: pre-processed data dict 
        N: number of samples, default set to 20
        history_feature_override: optional [P, 256] to skip encode+FSR (semantic Z+/Z-/rec).
        z1_override: optional [P*N, z_dim] to reuse the main-path latent.
        '''
        peds_stats = proc_data_dict['peds_stats']

        Zt = proc_data_dict['past_traj']
        # encode past trajectory; TrajFSR sits after History Encoder, before Update Decoder / sampler
        if history_feature_override is None:
            history_feature = self._apply_fsr(self.encode_history(proc_data_dict))
        else:
            history_feature = history_feature_override

        fut_traj = proc_data_dict['fut_traj']

        mus = []
        logvars = []

        dist_samples = []  # collect samples from network sampler

        ext_Zt = Zt.repeat(N, 1, 1)
        ext_history_feat = history_feature.repeat(N, 1)
        ext_fut_traj = fut_traj.repeat(N, 1, 1)

        pred_seq = []

        if vae_train:
            latent1 = self.encoder_future_eachstep(ext_fut_traj[:, :self.stride, :].contiguous().reshape(ext_fut_traj.size(0), -1))
            mu1 = latent1[:, :self.z_dim]
            logvar1 = latent1[:, self.z_dim:]
            mus.append(mu1)
            logvars.append(logvar1)

            var1 = logvar1.mul(0.5).exp_()
            eps1 = torch.FloatTensor(var1.size()).normal_()
            eps1 = eps1.cuda()
            z1 = eps1.mul(var1).add_(mu1)

        else:
            if noise:
                if z1_override is not None:
                    z1 = z1_override
                else:
                    samp = self.sample_past(history_feature).sigmoid().clamp(min=0.01, max=0.99)
                    samp = samp.reshape(history_feature.shape[0], -1, self.z_dim).permute(1, 0, 2)  # size: (N, Bs, z-dim)
                    samp = self.box_muller_transform(samp)
                    dist_samples.append(samp)

                    z1 = (samp * self.sigma1).reshape(-1, self.z_dim)
            else:
                z1 = torch.Tensor(ext_Zt.size(0), self.z_dim)
                z1.normal_(0, self.sigma1)
                z1 = z1.cuda()

        
        decoder_input = torch.cat((ext_history_feat, z1), dim=1)
        generated_xt1 = self.update_decoder(decoder_input).contiguous().reshape(ext_Zt.size(0), self.stride, 2)
        
        pred_seq.append(generated_xt1)

        # absolute position in state-configuration for edge potential
        abs_xt1 = generated_xt1 + ext_Zt[:, -1, :2].unsqueeze(1)
        edge_pairs, pair_dists, pair_diffs_xy = self.update_potential_edges(abs_xt1[:, -1, :].unsqueeze(1), peds_stats)

        xt1 = generated_xt1

        for t in range(self.stride, self.fut_step, self.stride):
            xt1_feature = self.encoder_space_state(xt1.contiguous().reshape(xt1.size(0), -1))

            if edge_pairs is not None:
                potential_ = torch.cat((pair_diffs_xy[:, 0].unsqueeze(1),
                                        pair_diffs_xy[:, 1].unsqueeze(1),
                                        1. / pair_dists.unsqueeze(1)), dim=1)

                feature_from_potential = self.edge_potential_net(potential_)
                
                indices = edge_pairs[:, 0].view(-1, 1).expand_as(feature_from_potential)
                xt1_feature.scatter_add_(0, indices, feature_from_potential)

            if vae_train:
                latent2 = self.encoder_future_eachstep(ext_fut_traj[:, t:t+self.stride, :].contiguous().reshape(ext_fut_traj.size(0), -1))
                mu2 = latent2[:, :self.z_dim]
                logvar2 = latent2[:, self.z_dim:]
                mus.append(mu2)
                logvars.append(logvar2)

                var2 = logvar2.mul(0.5).exp_()
                eps2 = torch.FloatTensor(var2.size()).normal_()
                eps2 = eps2.cuda()
                z2 = eps2.mul(var2).add_(mu2)

            else:
                if noise:
                    samp = self.sample_spacestate(xt1_feature).sigmoid().clamp(min=0.01, max=0.99)
                    samp = samp.reshape(N, history_feature.shape[0], -1)
                    samp = self.box_muller_transform(samp)       # size: (N, Bs, z-dim)
                    dist_samples.append(samp)

                    z2 = (samp * self.sigma2).reshape(-1, self.z_dim)
                else:
                    z2 = torch.Tensor(ext_Zt.size(0), self.z_dim)
                    z2.normal_(0, self.sigma2)
                    z2 = z2.cuda()

                
            delta_xt1plus_on_xt1 = self.dynamics_decoder(torch.cat((xt1_feature, z2), dim=1)).contiguous().reshape(ext_Zt.size(0), self.stride, 2)
            
            xt1plus_on_xt1 = xt1 + delta_xt1plus_on_xt1
            pred_seq.append(xt1plus_on_xt1)

            abs_xt1plus_on_xt1 = xt1plus_on_xt1 + ext_Zt[:, -1, :2].unsqueeze(1)
            # update edges for the next chunk
            edge_pairs, pair_dists, pair_diffs_xy = self.update_potential_edges(abs_xt1plus_on_xt1[:, -1, :].unsqueeze(1), peds_stats)
            
                
            xt1 = xt1plus_on_xt1
            abs_xt1 = abs_xt1plus_on_xt1


        mrf_sample = torch.cat(pred_seq, dim=1).contiguous().reshape(N, -1, fut_traj.shape[1], fut_traj.shape[2])

        if vae_train:
            return mrf_sample, mus, logvars
        
        if noise:
            return mrf_sample, dist_samples

        return mrf_sample
    

    def update_potential_edges(self, space_configure, peds_stats):
        edge_pair_list = []   # assume: [0] is agent itself
        dist_list = []
        diff_xy_list = []

        for i, peds_num in enumerate(peds_stats):
            srt_idx = sum(peds_stats[:i])
            end_idx = sum(peds_stats[:i+1])

            scene_coords = space_configure[srt_idx:end_idx].squeeze(1)
            diff_xy = scene_coords.unsqueeze(0) - scene_coords.unsqueeze(1)  # dim: N,N,2

            dist_to_nbor = torch.cdist(scene_coords.contiguous(), scene_coords.contiguous())

            pair_indices = ((dist_to_nbor <= 3.0) & (dist_to_nbor > 1e-4)).nonzero()
            dists = dist_to_nbor[pair_indices[:, 0], pair_indices[:, 1]]
            xydiffs = diff_xy[pair_indices[:, 0], pair_indices[:, 1], :]

            edge_pair_list.append(pair_indices + srt_idx)
            dist_list.append(dists)
            diff_xy_list.append(xydiffs)

        
        if len(edge_pair_list) > 0:
            new_edge_pair = torch.cat(edge_pair_list, dim=0).cuda().long()
            
            new_dist = torch.cat(dist_list, dim=0).cuda()
            new_diffs = torch.cat(diff_xy_list, dim=0).cuda()

            return new_edge_pair, new_dist, new_diffs
        else:
            return None, None, None
    

    def box_muller_transform(self, x: torch.FloatTensor):
        """Box-Muller transform, from paper NPSN"""
        shape = x.shape
        x = x.view(shape[:-1] + (-1, 2))
        z = torch.zeros_like(x, device=x.device)
        z[..., 0] = (-2 * x[..., 0].log()).sqrt() * (2 * np.pi * x[..., 1]).cos()
        z[..., 1] = (-2 * x[..., 0].log()).sqrt() * (2 * np.pi * x[..., 1]).sin()
        
        return z.view(shape)
            