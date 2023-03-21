import torch
import torch.nn as nn
import torch.autograd

import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import utils
from models.fcns import temporal_privacy_fcns

from tqdm import tqdm

class masdes(nn.Module):
    def __init__(self, args, logger):
        super(masdes, self).__init__()
        self.args = args
        self.logger = logger
        self.device = args.device
        self.model = temporal_privacy_fcns(args).cuda(self.device)     
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)
        
    def fetch_minibatch(self):
        Dt = np.zeros((self.args.batch_size, self.args.T_p, 1))  # B x N x 1
        DW = np.zeros((self.args.batch_size, self.args.T_p, self.args.D))  # B x N x D
        dt = self.args.T / (self.args.T_p - 1)
        Dt[:, 1:, :] = dt
        DW[:, 1:, :] = np.sqrt(dt) * np.random.normal(size=(self.args.batch_size, self.args.T_p - 1, self.args.D))

        t = np.cumsum(Dt, axis=1)  # B x N x 1
        W = np.cumsum(DW, axis=1)  # B x N x D
        t = torch.from_numpy(t).float().cuda(self.device)
        W = torch.from_numpy(W).float().cuda(self.device)
        return t, W

    def control_agents(self, X, M, t, temporal_privacy_index, player_index):  # M x 1, M x D
        Y, Z, Weight = self.model(X, M, t, temporal_privacy_index, player_index)  # M x 1
        return Y, Z, Weight

    def predict(self, x, mask, player_initial_time):
        X_list, W_list = self.SDE_simulation(x, mask, player_initial_time)
        return torch.stack(X_list, dim=0), torch.stack(W_list, dim=0)


    def Liouville(self, X0, b, sigma, dt, dWt):
        # X0    : B x D x 1
        # b     : B x D x 1
        # sigma : B x D x D
        
        # Simulate SDE with Euler-Maruyama first    
        
        X1_ = X0 + b * dt + torch.squeeze(
                                    torch.matmul(sigma, dWt.unsqueeze(-1)),
                                    dim=-1)
            
            
        # Score of transition density p(1, x_1 | 0, x_0)
        score = self.score_function(X1_, X0, b, sigma, dt)
        
        # if sigma : 
        drift_transformed = b - 0.5 * torch.squeeze(
                                            torch.matmul(sigma.pow(2), score.unsqueeze(-1)),
                                            dim=-1)
        
        # else:
        #     sigma_grad = torch.autograd(inputs = sigma.pow(2),,,,,,,,  )
        #     drift_transformed = b - 0.5 * ( sigma_grad + torch.squeeze(
        #                                                         torch.matmul(sigma.pow(2), score.unsqueeze(-1)),
        #                                                         dim=-1))
        
        
        return drift_transformed

    def score_function(self, X1, X0, b, sigma, dt):
        # X1    : B x D x 1
        # X0    : B x D x 1
        # b     : B x D x 1
        # sigma : B x D x D
        Sigma = sigma * sigma  # sigma : dianonal
        InvSigma = torch.diag_embed(1 / torch.diagonal(Sigma, dim1=-2,dim2=-1))
        
        # \nabla \log p_t(x) = -(dt)^{-1} * Sigma^{-1} * (X1 - X0 - b*dt)
        
        score = -(1/dt) * torch.squeeze(
            torch.matmul(
                InvSigma, (X1 - X0 - b*dt).unsqueeze(-1)),
                dim=-1
            )
        
        return score.detach()
        
    # Simulate forward controlled stochastic differential equation.
    def SDE_simulation(self, X, mask, player_index):
        player_initial_time = player_index
        X_list = []
        W_list = []       
        

        t, W = self.fetch_minibatch()
        
        X0 = X.clone()
        t0 = t[:, player_initial_time, :]
        W0 = W[:, player_initial_time, :]
        M0 = mask[:, player_initial_time, :]

        # X_list.append(X0)
        Y0, Z0, Weight0 = self.control_agents(X0, M0, t0, 0, player_initial_time)  # B x 1, B x D
        

        for t_gap in range(1, self.T_p - player_initial_time):
            t1 = t[:, player_initial_time + t_gap, :]
            W1 = W[:, player_initial_time + t_gap, :]
            M1 = mask[:, player_initial_time + t_gap, :]
            
            dt = t1 - t0
            dWt = W1 - W0
            drift_ = self.drift(t0, X0, Z0)
            diffusion_ = self.diffusion(t0, X0, Y0)
            
            drift_ode = self.Liouville(X0, drift_, diffusion_, dt, dWt)


            ##### Liouville Equation #####
            X1 = (
                X0 
                + drift_ode * dt
            )
            
            # X1 = X0 + drift_ * dt + torch.squeeze(
            #                             torch.matmul(diffusion_, dWt.unsqueeze(-1)),
            #                             dim=-1)

            t0 = t1
            W0 = W1
            X0 = X1
            M0 = M1
            
            Y0, Z0, Weight0 = self.control_agents(X0, M0, t0, player_initial_time + t_gap, player_initial_time) 
            

            if t_gap + player_initial_time >= self.T_o:
                X_list.append(X1)
                W_list.append(Weight0)

        return X_list, W_list
    
    def fictitious_play(self, t, W, Y, M, label, stage):
        target_Y = Y[:, self.T_o:, :]
        mask = M[:, self.T_o:, :]
        
        total_decision = torch.zeros(self.A, self.PI, self.B, self.D, requires_grad=True).cuda()
        total_weight = torch.zeros(self.A, self.PI, self.B, 1, requires_grad=True).cuda()

        for N in range(self.A):
            Y_N = Y[:, N, :]
            X_t, rho_t = self.predict(Y_N, M, N)
            total_decision[N, :, :, :] += X_t
            total_weight[N,:,:,:] += rho_t
        
        loss_train = 0.
        loss_mse = 0.        
        loss = 0.
     
        ################### Fictitious Play ######################
    
        for A_ in tqdm(range(self.A), desc='Agent', mininterval=0.1):    
            ## Simulated Decisions from previous agents
            TD_dummy = total_decision.clone().detach() 
            TW_dummy = total_weight.clone().detach()
            
            Y_A = Y[:, A_, :]
            decision_A, weight_A = self.predict(Y_A, M, A_)  # 1 x PI x B, D   #\alpha^{N}  ``gradient on''
            
            decision_A, weight_A = decision_A.unsqueeze(0), weight_A.unsqueeze(0)
            with torch.no_grad():
                decision_not_A = torch.cat([TD_dummy[:A_, :, :, :], TD_dummy[A_+1:, :, :]]) # A-1 X PI X B X D  #\alpha^{-N}  ``gradient off''
                weight_not_A = torch.cat([TW_dummy[:A_, :, :, :], TW_dummy[A_+1:, :, :]]) # A-1 X PI X B X 1  #\alpha^{-N}  ``gradient off''
            
            total_decision_A = torch.cat((decision_A, decision_not_A), dim=0)
            total_weight_A = torch.cat((weight_A, weight_not_A), dim=0)      
            
            total_weight_A = F.softmax(total_weight_A, dim=0)
            
            prediction = (total_decision_A * total_weight_A).sum(0)

            running_cost = self.running_cost(target_Y, prediction.transpose(0,1), mask)
            terminal_cost = self.terminal_cost(decision_A, weight_A)
            
            loss_train += (running_cost + terminal_cost).item()  
            loss += (running_cost + terminal_cost) 
            
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        loss_train /= (self.args.A)
        return loss_train
    
    
    def train_and_eval(self, train_loader, test_loader):
        mse_best = 1e5
        nll_best = -1e5
        
        for stage in range(self.args.n_stages):
            losses_train = self.train(train_loader)
            mse_test, nll_test = self.evaluation(test_loader)


    def train(self, train_loader):
        losses_train = 0.
        for i, (data, mask) in enumerate(train_loader):
            train_data, train_mask = data.to(self.device), mask.to(self.device)
            t_batch ,W_batch = self.fetch_minibatch()
            loss_train = self.fictitious_play(t_batch, W_batch, train_data, train_mask, None, stage)        
            msg = ("Iter = [%d/%d], Train_Loss: : %.4f" % (i+1, len(train_loader), 100*loss_train))
            self.logger.info(msg)    
            
            losses_train += losses_train
        return losses_train / len(train_loader)
    
    
    ######
    def evaluation(self, test_loader):
        with torch.no_grad():
            for i, (data, mask) in enumerate(test_loader):
                test_data, test_mask = data.to(self.device), mask.to(self.device)
                
                target_Y = test_data[:, self.args.T_o:, :]
                target_mask = test_mask[:, self.args.T_o:, :]
                
                total_decision = torch.zeros(self.args.A, self.args.T_p - self.args.T_o, self.args.batch_size, self.args.D).to(self.device)
                total_aggregation = torch.zeros(self.args.A, self.args.T_p - self.args.T_o, self.args.batch_size, 1).to(self.device)                
                
                for I_ in range(self.args.A):
                    Y_I = test_data[:, I_, :]
                    X_t, A_t = self.predict(Y_I, test_mask, I_)
                    total_decision[I_, :, : ,:] += X_t
                    total_aggregation[I_, :, :, :] += A_t
                    
                total_aggregation = F.softmax(total_aggregation, dim = 0)
                
                Lambda = (total_decision * total_aggregation).sum(0)
                
                
    def running_cost(self, target_Y, predict_Y, mask):
        loss_running = utils.get_mse(target_Y, predict_Y, mask)
        return loss_running
    
    def terminal_cost(self, state_A, weight_A):  # B x 1
        # loss_terminal = weight_A.pow(2).mean()
        return 0.

    def drift(self, t, X, Z):  # B x 1, B x D, B x D
        if self.sde_type == "mckean_vlasov":
            out = torch.mean(Z, dim=0).unsqueeze(0) - Z
        elif self.sde_type == "vanilla":
            out = Z

        return out  # B x 3 x D

    # Non-degenerate type diffusion function
    def diffusion(self, t, X, Y):  # B x 1, B x D, B x 1
        Y = torch.clamp(Y, min=self.sigma_low, max=self.sigma_low)
        return torch.diag_embed(Y).to(self.device)  # B x D x D
