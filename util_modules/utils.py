import json
import os
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from util_modules import Metrics, utils
from sklearn.decomposition import PCA
import torch.nn.functional as F
import warnings
warnings.simplefilter("ignore", UserWarning)
# Set plt params
sns.set(style='ticks', font_scale=1.2)
plt.rcParams['figure.figsize'] = 12,8

def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_args(params, filename, path):
    print("saved the new arguments")
    path_ = os.path.join(path, filename)
    with open(path_+".json", 'w') as f:
        json.dump(vars(params), f, indent=4)
        
def save_model(model_, path_, name):
    model_.train(False)
    torch.save(model_.state_dict(), os.path.join(path_, f"{name}_cp.pth"))

def load_model(model, path_, name):
    try:
        if os.path.exists(os.path.join(path_, f'{name}_cp.pth')):
            print(f"Pre-trained model ({name}) is loaded")
            state_dict = torch.load(os.path.join(path_, f'{name}_cp.pth'), weights_only=True)
            model.load_state_dict(state_dict, strict=True)
            # model = model.load_state_dict(torch.load(
            #         os.path.join(path_, f'{name}_cp.pth'),weights_only=True))
            return model, False
        else:
            print("No pre-trained model exists")
            return model, True
    except: 
        raise ImportError("Loading trained model failed. Check if the model parameters and arguments are matching")
        return None, None

def onehot_encoding(c, classes):
    N, L, _ = c.shape
    return torch.nn.functional.one_hot(c.squeeze(dim=-1).to(torch.long), num_classes=classes)

def zeroout_gradient(layers_):
    # Zero out the gradients in layer b computed with respect to out_B 
    for net in layers_:
        for param in net.parameters():
            if param.grad == None:
                pass
            else:
                param.grad = torch.zeros_like(param.grad)


class Evaluation():
    def __init__(self, plot_path, 
                 hist_path,
                 logger):
        super(Evaluation, self).__init__()
                #Early stopping 
        self.latent_pics = os.path.join(plot_path, 'latent_pics')
        utils.mkdir(self.latent_pics)
        self.code_pics = os.path.join(plot_path, 'code_pics')
        utils.mkdir(self.code_pics)
        self.data_pics = os.path.join(plot_path, 'data_pics')
        utils.mkdir(self.data_pics)
        self.max_sample = -1 # upper limit for number of samples in computation 
        self.logger = logger
    def qualitative_analysis(self, 
                            prior_z, z_h, z_E, z0_E,
                            prior_c, c_E, c_gt, c_Q,
                            time_ind,
                            discrete = False,
                            epoch = "",
                            prior_c2 = None, c_E2 = None, c_Q2 = None,
                            
                            prior_zG = None, Z0G = None
                            ):
        '''
        Yield the following visual (qualitative) analysis
        The c inputs are logits. 
        
        
        1. z_h vs prior_z --> show what the flow transform learns vs the prior distribution p(z').
        
        2. z_h vs z_E --> show how p_h(z) and q_E(z|x) matches (prior VS posterior).
        
        3. z_E with time index to see temporal structure.
        
        4. XX (Estimating this is not possible) z_E with (i) groud truth code (c_gt) and
           (ii) inferred code (c_E) --> we expect to see "NO structure" in this since z_E 
           is meant to be "invariant to the code c".
        
        5. prior_c vs c_E
        
        6. c_E labelled by c_gt --> there will be class-id mismatch, since we are doing 
           unsupervised learning (i.e., clustering with f_C and Q nets).
        
        '''
        # print(prior_z.shape)
        # print(z_h.shape)
        # print(z_E.shape)
        
        # print(prior_c.shape)
        # print(c_E.shape)
        # print(c_gt.shape)
        # print(time_ind.shape)
        
        # print("prior_c in eval", prior_c.mean())
        # print("C_E in eval", c_E.mean())
        # print("C_Q in eval", c_Q.mean())
        self.logger.info(f"Computing plots (epoch = {epoch}) ... ")
        # Latent represenataions
        
        # Normalize z_h and z_E for comparison with the prior (mean 0 std 1) # TODO noramlize as a whole
        # z_h = (z_h - z_h.mean(dim = 0, keepdim = True)) / torch.clamp(z_h.std(dim = 0, keepdim = True), min=1e-6) 
        # z_E = (z_E - z_E.mean(dim = 0, keepdim = True)) / torch.clamp(z_E.std(dim = 0, keepdim = True), min=1e-6) 

        # PCA on z
        prior_z, z_h, z_E, z0_E = prior_z.detach().cpu().numpy(), z_h.detach().cpu().numpy(), z_E.detach().cpu().numpy(), z0_E.detach().cpu().numpy()
        pca = PCA(n_components=2)
        pca.fit(np.concatenate((prior_z, z_h), axis = 0))
        prior_z, z_h = pca.transform(prior_z), pca.transform(z_h)

        pca = PCA(n_components=2)
        pca.fit(np.concatenate((z_E, z0_E), axis = 0))
        z_E, z0_E = pca.transform(z_E), pca.transform(z0_E)
        # prior_z, z_h, z_E = pca.transform(prior_z), pca.transform(z_h), pca.transform(z_E)
        
        # pca.fit(z_E)
        # z_E = pca.transform(z_E)
        if prior_zG is not None:
            prior_zG, Z0G = prior_zG.detach().cpu().numpy(), Z0G.detach().cpu().numpy()
            pca = PCA(n_components=2)
            pca.fit(np.concatenate((prior_zG, Z0G), axis = 0))
            prior_zG, Z0G = pca.transform(prior_zG), pca.transform(Z0G)
            # 00) 
            self.comparison_plot(prior_zG, Z0G, labels = [r"$p(z0)$", r"$q_{E}(z0|x_{G})$"], path = self.latent_pics, plotname = "Gen_prior_vs_posterior", epo = epoch)
            
        # 0)
        self.comparison_plot(z_E, z0_E, labels = [r"$q_{h}(z|x)$", r"$q_{E}(z0|x)$"], path = self.latent_pics, plotname = "Posterior_zh_vs_z0", epo = epoch)
        self.kde_plot(z_E, z0_E, labels = [r"$q_{h}(z|x)$", r"$q_{E}(z0|x)$"], path = self.latent_pics, plotname = "Posterior_kde_zh_vs_z0", epo = epoch)
        
        # 1)
        self.comparison_plot(z_h, prior_z, labels = [r"$z \sim p_{h}(z|x)$", r"$z' \sim p(z0|x)$"], path = self.latent_pics, plotname = "overgaussian_zh_vs_priorZ", epo = epoch)
        self.kde_plot(z_h, prior_z, labels = [r"$z \sim p_{h}(z|x)$", r"$z' \sim p(z0|x)$"], path = self.latent_pics, plotname = "overgaussian_kde_zh_vs_priorZ", epo = epoch)

        # 2)
        # self.comparison_plot(z_h, z_E, labels = [r"$z \sim p_{h}(z)$", r"$z \sim q(z|x)$"], path = self.latent_pics, plotname = "Sampled_zh_vs_posteriorZ", epo = epoch)
        # self.kde_plot(z_h, z_E, labels = [r"$z \sim p_{h}(z)$", r"$z \sim q(z|x)$"], path = self.latent_pics, plotname = "kde_Sampled_zh_vs_posteriorZ", epo = epoch)
        
        # 3)
        # self.structural_plot(z_h, time_ind.squeeze().detach().cpu().numpy(), aux_name = "Time", path = self.latent_pics, plotname = "Sampled_ZH_with_time", epo = epoch)
        self.structural_plot(z_E, time_ind.squeeze().detach().cpu().numpy(), aux_name = "Time", path = self.latent_pics, plotname = "posteriorZ_X", epo = epoch)
        self.structural_plot(z0_E, time_ind.squeeze().detach().cpu().numpy(), aux_name = "Time", path = self.latent_pics, plotname = "posteriorZ0_X", epo = epoch)

        self.structural_plot(prior_z, time_ind.squeeze().detach().cpu().numpy(), aux_name = "Time", path = self.latent_pics, plotname = "overgaussian_Z0_X", epo = epoch)
        self.structural_plot(z_h, time_ind.squeeze().detach().cpu().numpy(), aux_name = "Time", path = self.latent_pics, plotname = "overgaussian_Z_X", epo = epoch)
        # Code represenataions
        if c_E.shape[-1] >= 2:
            # c_Q = (c_Q - c_Q.mean(dim = 0, keepdim = True)) / torch.clamp(c_Q.std(dim = 0, keepdim = True), min=1e-6) 
            # c_E = (c_E - c_E.mean(dim = 0, keepdim = True)) / torch.clamp(c_E.std(dim = 0, keepdim = True), min=1e-6) 
            if discrete:
                # Get the categorical labels of p(c)
                # prior_c_labels = torch.argmax(F.softmax(prior_c[:,-1,:], dim = -1), dim = -1).squeeze().detach().cpu().numpy() # (N,)
                # prior_c_labels = torch.argmax(F.softmax(prior_c[:,-1,:], dim = -1), dim = -1).squeeze().detach().cpu().numpy() # (N,)
                prior_c_labels = torch.argmax(prior_c[:,-1,:], dim = -1).squeeze().detach().cpu().numpy()
            else: 
                prior_c_labels = None
            # TODO in case of prior_c is discrete
            # Consider the last position only since sliding window is applied (i.e., the rest will be redundant)
            # prior_c, c_E, c_Q = prior_c.detach().cpu().numpy()[:,-1,:], c_E.detach().cpu().numpy()[:,-1,:], c_Q.detach().cpu().numpy()[:,-1,:] # (N,dc)
            prior_c, c_E, c_Q = prior_c.detach().cpu().numpy()[:,-1,:] if not discrete else None, c_E.detach().cpu().numpy()[:,-1,:], c_Q.detach().cpu().numpy()[:,-1,:] # (N,dc)

            # PCA on C
            if (c_E.shape[-1] > 2):
                pca = PCA(n_components=2)
                if discrete:
                    pca.fit(np.concatenate((c_E, c_Q), axis = 0))
                    # prior_c, c_E, c_Q = pca.transform(prior_c), pca.transform(c_E), pca.transform(c_Q) # (N,2)
                    prior_c, c_E, c_Q = None, pca.transform(c_E), pca.transform(c_Q) # (N,2)
                else:
                    pca.fit(np.concatenate((c_E, c_Q), axis = 0))
                    prior_c, c_E, c_Q = pca.transform(prior_c), pca.transform(c_E), pca.transform(c_Q) # (N,2)
            if not discrete:
                # 5)
                self.comparison_plot(prior_c, c_E, aux = prior_c_labels, labels = [r"$p(c)$", r"$q_{C}(c|x)$"], path = self.code_pics, plotname = "pc vs posteriorC", epo = epoch)
                # 5.2)
                self.comparison_plot(prior_c, c_Q, aux = prior_c_labels, labels = [r"$p(c)$", r"$q_{Q}(c|x)$"], path = self.code_pics, plotname = "pc vs posteriorC_Q", epo = epoch)
                # 6)
            c_gt = c_gt[:,-1,:]
            self.structural_plot(c_E, c_gt.squeeze().detach().cpu().numpy(), aux_name = "Operating conditions", path = self.code_pics, plotname = "posteriorC_with_true_ocs", epo = epoch)
            if prior_c_labels is not None:
                self.structural_plot(c_Q, prior_c_labels, aux_name = "Operating conditions", path = self.code_pics, plotname = "posteriorQ_with_priors", epo = epoch)

        elif prior_c.shape[-1] == 1:
            # continous of dim 1 case -> line plot
            if not discrete:
                prior_c, c_E, c_Q = prior_c.detach().cpu().numpy()[:,-1,:] if not discrete else None, c_E.detach().cpu().numpy()[:,-1,:], c_Q.detach().cpu().numpy()[:,-1,:] # (N,dc)
                prior_c_labels = None
                # 5)        
                self.comparison_plot(prior_c, c_E, aux = prior_c_labels, labels = [r"$p(c)$", r"$q_{C}(c|x)$"], path = self.code_pics, plotname = "pc vs posteriorC", epo = epoch)
                # 5.2)
                self.comparison_plot(prior_c, c_Q, aux = prior_c_labels, labels = [r"$p(c)$", r"$q_{Q}(c|x)$"], path = self.code_pics, plotname = "pc vs posteriorC_Q", epo = epoch)
                # 6)
                c_gt = c_gt[:,-1,:]
                self.structural_plot(c_E, c_gt.squeeze().detach().cpu().numpy(), aux_name = "Operating conditions", path = self.code_pics, plotname = "posteriorC_with_true_ocs", epo = epoch)

        if prior_c2 is not None:
            # Always discrete
            prior_c_labels2 = torch.argmax(prior_c2[:,:], dim = -1).squeeze().detach().cpu().numpy()
            c_E2_labels2 = torch.argmax(c_E2[:,:], dim = -1).squeeze().detach().cpu().numpy()
            c_E2, c_Q2 = c_E2.detach().cpu().numpy()[:,:], c_Q2.detach().cpu().numpy()[:,:] # (N,dc)
            if (c_E2.shape[-1] > 2):
                pca = PCA(n_components=2)
                if discrete:
                    pca.fit(np.concatenate((c_E2, c_Q2), axis = 0))
                    c_E2, c_Q2 =  pca.transform(c_E2), pca.transform(c_Q2) 
          
            self.structural_plot(c_Q2, prior_c_labels2, aux_name = "Operating conditions", path = self.code_pics, plotname = "HC_posteriorQ2_with_priors", epo = epoch)
            self.structural_plot(z_E, c_E2_labels2, aux_name = "Health condition", path = self.latent_pics, plotname = "HC_posteriorZ_X", epo = epoch)
            self.structural_plot(z0_E, c_E2_labels2, aux_name = "Health condition", path = self.latent_pics, plotname = "HC_posteriorZ0_X", epo = epoch)
    def eval_code(self, c_E, C_Q, 
                  c_ground_truth,
                  epoch = None):
        '''
        Evaluate inference of discrete codes of learned f_C and Q on "real data" using a clustering acc and purity metric.
        The inputs must be (or approximately) discrete already
        1. c_E vs c_gt 
        
        2. C_Q vs c_gp
        '''
        self.logger.info(f"Evaluating code on clustering metrics (epoch:{epoch})... ")
        Metrics.clustering_acc
        Metrics.clustering_purity

    def eval_rul(self, rul, rul_pred,
                 epoch = ""):
        self.logger.info(f"Evaluating RUL (epoch:{epoch})... ")
        
    def vis_learned_time_embedding(self, time_embedding, time_ind,
                                    epoch = ""):
        time_embedding = time_embedding.squeeze(1)
        self.logger.info(f"Visualizing The learned time embedding (epoch:{epoch})... ")
        time_embedding = time_embedding.squeeze().detach().cpu().numpy()
        pca = PCA(n_components=2)
        pca.fit(time_embedding)
        # prior_c, c_E, c_Q = pca.transform(prior_c), pca.transform(c_E), pca.transform(c_Q) # (N,2)
        time_embedding  = pca.transform(time_embedding) # (N,2)
        self.structural_plot(time_embedding, time_ind.squeeze().detach().cpu().numpy(), aux_name = "Time", path = self.latent_pics, plotname = "Time embeddings", epo = epoch)

    def comparison_plot(self, x1, x2 = None, aux = None,
                        labels = [], path= '', plotname = '', epo = ''):
        fig = plt.figure(figsize=(12, 8), 
            dpi = 600) 
        axes = fig.subplots()
        if x1.shape[-1] == 1:
            temp = np.zeros_like(x1)
            x1 = np.concatenate((x1,temp), axis = 1)
        if x2.shape[-1] == 1:
            temp = np.zeros_like(x1)
            x2 = np.concatenate((x2,temp), axis = 1)
            
        if  aux is None:
            axes.scatter(x = x1[:,0], y= x1[:,1], 
                    s=30, color="red" , cmap="Spectral", alpha = 0.5, label = labels[0]) # , edgecolors= "black"
        else:
            axes.scatter(x = x1[:,0], y= x1[:,1], 
                    s=30, c=aux[:], cmap="Spectral", alpha = 0.5, label = labels[0]) # , edgecolors= "black"
        scatter = axes.scatter(x = x2[:,0], y= x2[:,1], 
                    s=30, color="blue", cmap="Spectral", alpha = 0.5, label = labels[1])
        
        if aux is not None:
            cbar = plt.colorbar(scatter)
            cbar.set_label("Operating conditions", fontweight='bold', fontsize=20)
        else:
            legend = axes.legend(fontsize=20, loc='upper center', bbox_to_anchor=(0.5, 1.1)) # TODO
        plt.xlabel("Principal component 1", fontsize=20, fontweight='bold')
        plt.ylabel("Principal component 2", fontsize=20, fontweight='bold')
        plt.xticks(fontweight='bold', fontsize = 20)   
        plt.yticks(fontweight='bold', fontsize = 20)
        plt.tight_layout()
        plt.savefig(os.path.join(path,plotname+f"_{epo}.png" )) 
        plt.clf()   
        plt.close(fig)
        
    def structural_plot(self, x1, aux ,
                        aux_name="", path= '', plotname = '', epo = ''):
        if x1.shape[-1] == 1:
            temp = np.zeros_like(x1)
            x1 = np.concatenate((x1,temp), axis = 1)
        fig = plt.figure(figsize=(12, 8), 
            dpi = 600) 
        axes = fig.subplots()
        
        scatter = axes.scatter(x = x1[:,0], y= x1[:,1], c =aux[:], 
                                s=20, cmap="Spectral", alpha = 0.8, edgecolors= "black")
        
        # # Get unique class labels
        # unique_classes = np.unique(y[:, 0])

        # # Iterate through each class and mark one point
        # for ii, class_label in enumerate(unique_classes):
        #     class_indices = np.where(y[:, 0] == class_label)[0]
        #     sample_index = class_indices[10]  # Choose the first sample for marking
        #     axes.text(x[sample_index, 0], x[sample_index, 1], f'{ii}', color='black', fontsize=15, 
        #               ha='center', va='center', fontweight='bold')            
        cbar = plt.colorbar(scatter)
        cbar.set_label(aux_name, fontweight='bold', fontsize=20)

        # cbar.set_label(r'$\mathbf{c}$', fontweight='bold', fontsize=20)
        
        plt.xlabel("Principal component 1", fontsize=20, fontweight='bold')
        plt.ylabel("Principal component 2", fontsize=20, fontweight='bold')
        plt.xticks(fontweight='bold', fontsize = 20)   
        plt.yticks(fontweight='bold', fontsize = 20)
        plt.tight_layout()
        plt.savefig(os.path.join(path,plotname+f"_{epo}.png" )) 
        plt.clf()   
        plt.close(fig)
    
    def kde_plot(self, x1, x2 = None, 
                        labels = [], path= '', plotname = '', epo = ''):
        
        for ii in range(x1.shape[1]):
            fig = plt.figure(figsize=(12, 8), 
                dpi = 600) 
            axes = fig.subplots()
            sns.kdeplot(x1[:, ii], label=labels[0], ax=axes, fill=True, alpha=0.4, color="blue")
            sns.kdeplot(x2[:, ii], label=labels[1], ax=axes, fill=True, alpha=0.4, color="red")

            legend = axes.legend(fontsize=20, loc='upper center', bbox_to_anchor=(0.5, 1.1)) # TODO
            plt.xlabel("Principal component 1", fontsize=20, fontweight='bold')
            plt.ylabel("Value", fontsize=20, fontweight='bold')
            plt.xticks(fontweight='bold', fontsize = 20)   
            plt.yticks(fontweight='bold', fontsize = 20)
            plt.tight_layout()
            plt.savefig(os.path.join(path,plotname+f"dim({ii})_{epo}.png" )) 
            plt.clf()   
            plt.close(fig)
        
    def line_plot(self, x, y = None, label = "", inst = 1, instant = 0):
        N = len(x)
        if instant == 0:
            if inst > N:
                raise ValueError("S must be less than or equal to N to ensure unique values.")
            indx = random.sample(range(1, N + 1), inst)
        
        else:
            indx = [instant]
    
        for i in indx:
            X = x[i]
            for c in range(X.shape[1]):
                fig = plt.figure(figsize=(12, 8), 
                                dpi = 600) 
                axes = fig.subplots()
                axes.plot(X[:,c], c = "blue", linewidth=1, alpha=1, label = label)
                
                plt.xlabel("Time", fontsize=20, fontweight='bold')
                plt.ylabel("Magnitude", fontsize=20, fontweight='bold')
                plt.xticks(fontweight='bold', fontsize = 20)   
                plt.yticks(fontweight='bold', fontsize = 20)
                legend = axes.legend(fontsize=20, loc='upper center', bbox_to_anchor=(0.5, 1.1))
                plt.tight_layout()
                plt.savefig(os.path.join(self.data_pics, f"{label}_machine{i}_channel_{c}.png" ))    
                plt.clf()   
                plt.close(fig)

                if c > 0:
                    break;
    def recon_plot(self, x, y, label = ["",""], epoch = "", title = ""):
        x= x[:,:2].detach().cpu().numpy()
        y = y[:,:2].detach().cpu().numpy()
        
        for c in range(x.shape[-1]):
            fig = plt.figure(figsize=(12, 8), 
                            dpi = 600) 
            axes = fig.subplots()
            axes.plot(x[:,c], c = "blue", linewidth=1, alpha=1, label = label[0])
            axes.plot(y[:,c], c = "red", linewidth=1, alpha=0.7, ls = "--", label = label[1])

            plt.xlabel("Time", fontsize=20, fontweight='bold')
            plt.ylabel("Magnitude", fontsize=20, fontweight='bold')
            plt.xticks(fontweight='bold', fontsize = 20)   
            plt.yticks(fontweight='bold', fontsize = 20)
            legend = axes.legend(fontsize=20, loc='upper center', bbox_to_anchor=(0.5, 1.1))
            plt.tight_layout()
            plt.savefig(os.path.join(self.data_pics, f"{title}_channel_{c}_epoch{epoch}.png" ))    
            plt.clf()   
            plt.close(fig)

            if c > 6:
                break;
    def forecasting_plot(self, x, pred, y, label = ["",""], epoch = "", title = ""):
        W, T, dx = x.shape
        H, T, dx = pred.shape
        # TODO Check how to plot this from the original ocir code  
        pred_x = torch.concat((x.view(-1,dx), pred.view(-1,dx)), dim = 0).detach().cpu().numpy() # Stationarized 
        y_x = torch.concat((x.view(-1,dx), y.view(-1,dx)), dim = 0).detach().cpu().numpy() # non-stationary
        
        for c in range(x.shape[-1]):
            fig = plt.figure(figsize=(12, 8), 
                            dpi = 600) 
            axes = fig.subplots()
            axes.plot(x[:,c], c = "blue", linewidth=1, alpha=1, label = label[0])
            axes.plot(y[:,c], c = "red", linewidth=1, alpha=0.7, ls = "--", label = label[1])

            plt.xlabel("Time", fontsize=20, fontweight='bold')
            plt.ylabel("Magnitude", fontsize=20, fontweight='bold')
            plt.xticks(fontweight='bold', fontsize = 20)   
            plt.yticks(fontweight='bold', fontsize = 20)
            legend = axes.legend(fontsize=20, loc='upper center', bbox_to_anchor=(0.5, 1.1))
            plt.tight_layout()
            plt.savefig(os.path.join(self.data_pics, f"{title}_channel_{c}_epoch{epoch}.png" ))    
            plt.clf()   
            plt.close(fig)

            if c > 6:
                break;
    
    def info_qualitative_analysis(self, 
                            prior_z, z_h, z_E,
                            prior_c, c_E, c_gt, c_Q,
                            time_ind,
                            discrete = False,
                            discreteuniform = True,
                            epoch = ""):
        '''
        Yield the following visual (qualitative) analysis
        The c inputs are logits. 
        InfoGAN
        
        '''
        # # print(prior_z.shape)
        # # print(z_h.shape)
        # # print(z_E.shape)
        
        # # print(prior_c.shape)
        # # print(c_E.shape)
        # # print(c_gt.shape)
        # # print(time_ind.shape)
        
        # print("prior_c in eval", prior_c.mean())
        # print("C_E in eval", c_E.mean())
        # print("C_Q in eval", c_Q.mean())
        
        if z_h is not None:
            self.logger.info(f"Computing plots (epoch = {epoch}) ... ")
            # Latent represenataions
            # Normalize z_h for comparison with the prior (mean 0 std 1)
            z_h = (z_h - z_h.mean(dim = 0, keepdim = True)) / torch.clamp(z_h.std(dim = 0, keepdim = True), min=1e-6) 
            # PCA on z
            prior_z, z_h = prior_z.detach().cpu().numpy(), z_h.detach().cpu().numpy()
            pca = PCA(n_components=2)
            pca.fit(np.concatenate((prior_z, z_h), axis = 0))
            prior_z, z_h = pca.transform(prior_z), pca.transform(z_h)
            
            # 1)
            self.comparison_plot(z_h, prior_z, labels = [r"$p_{h}(z)$", r"$p(z')$"], path = self.latent_pics, plotname = "zh_vs_priorZ", epo = epoch)
            # 2)
            # self.comparison_plot(z_h, z_E, labels = [r"$p_{h}(z)$", r"$q_{E}(z|x)$"], path = self.latent_pics, plotname = "zh_vs_posteriorZ", epo = epoch)
            
            # 3)
            # self.structural_plot(z_h, time_ind.squeeze().detach().cpu().numpy(), aux_name = "Time", path = self.latent_pics, plotname = "posteriorZ_with_time", epo = epoch)

        # Code represenataions
        if not discreteuniform:
            if prior_c.shape[-1] >= 2:
                # c_Q = (c_Q - c_Q.mean(dim = 0, keepdim = True)) / torch.clamp(c_Q.std(dim = 0, keepdim = True), min=1e-6) 

                if discrete:
                    # Get the categorical labels of p(c)
                    prior_c_labels = torch.argmax(F.softmax(prior_c[:,-1,:], dim = -1), dim = -1).squeeze().detach().cpu().numpy() # (N,)
                else: prior_c_labels = None
                # Consider the last position only since sliding window is applied (i.e., the rest will be redundant)
                prior_c,  c_Q = prior_c.detach().cpu().numpy()[:,-1,:],  c_Q.detach().cpu().numpy()[:,-1,:] # (N,dc)
                
                # PCA on C
                if not (prior_c.shape[-1] == 2):
                    pca = PCA(n_components=2)
                    pca.fit(np.concatenate((prior_c, c_Q), axis = 0))
                    prior_c, c_Q = pca.transform(prior_c), pca.transform(c_Q) # (N,2)
            
                # 5)
                # self.comparison_plot(prior_c, c_E, aux = prior_c_labels, labels = [r"$p(c)$", r"$q_{C}(c|x)$"], path = self.code_pics, plotname = "pc vs posteriorC", epo = epoch)
                # 5.2)
                self.comparison_plot(prior_c, c_Q, aux = prior_c_labels, labels = [r"$p(c)$", r"$q_{Q}(c|x)$"], path = self.code_pics, plotname = "pc vs posteriorC_Q", epo = epoch)
                
                # 6)
                c_gt = c_gt[:,-1,:]
                if prior_c_labels is not None:
                    self.structural_plot(c_Q, prior_c_labels, aux_name = "True operating conditions", path = self.code_pics, plotname = "posteriorQ_with_priors", epo = epoch)
            elif prior_c.shape[-1] == 1:
                # continous of dim 1 case -> line plot
                pass
                raise NotImplementedError("")
        else:
            prior_c_labels = torch.argmax(prior_c[:,-1,:], dim = -1).squeeze().detach().cpu().numpy() # (N,)
            # PCA on C
            c_Q = c_Q.detach().cpu().numpy()[:,-1,:] # (N,dc)
            if not (prior_c.shape[-1] == 2):
                pca = PCA(n_components=2)
                pca.fit(c_Q)
                c_Q =pca.transform(c_Q) # (N,2)
                
            self.structural_plot(c_Q, prior_c_labels, aux_name = "True operating conditions", path = self.code_pics, plotname = "posteriorQ_with_priors", epo = epoch)


    def nfvae_qualitative_analysis(self, 
                            z0, z_h, z_E, z_h_z0,
                            time_ind,
                            discrete = False,
                            epoch = ""):
        '''
        VAE
        '''
        self.logger.info(f"Computing plots (epoch = {epoch}) ... ")
        # Latent represenataions
        # Normalize z_h for comparison with the prior (mean 0 std 1)
        # if z_h is not None:
        #     z_h = (z_h - z_h.mean(dim = 0, keepdim = True)) / torch.clamp(z_h.std(dim = 0, keepdim = True), min=1e-6) 
        # z_E = (z_E - z_E.mean(dim = 0, keepdim = True)) / torch.clamp(z_E.std(dim = 0, keepdim = True), min=1e-6) 
        # z0 = (z0 - z0.mean(dim = 0, keepdim = True)) / torch.clamp(z0.std(dim = 0, keepdim = True), min=1e-6) 
        # z_h_z0 = (z_h_z0 - z_h_z0.mean(dim = 0, keepdim = True)) / torch.clamp(z_h_z0.std(dim = 0, keepdim = True), min=1e-6) 

        # PCA on z
        if z_h is not None:
            z_h = z_h.detach().cpu().numpy()
        z_E = z_E.detach().cpu().numpy()
        z0 = z0.detach().cpu().numpy()
        z_h_z0 = z_h_z0.detach().cpu().numpy()
        
        pca = PCA(n_components=2)
        if z_h is not None:
            pca.fit(np.concatenate((z_E, z_h), axis = 0))
            z_E, z_h= pca.transform(z_E), pca.transform(z_h)
            pca.fit(np.concatenate((z0, z_h_z0), axis = 0))
            z0, z_h_z0 = pca.transform(z0), pca.transform(z_h_z0)
        else:
            pca.fit(z_E)
            z_E, z0, z_h_z0= pca.transform(z_E), pca.transform(z0), pca.transform(z_h_z0)     
        # 1)
        if z_h is not None:
            self.comparison_plot(z_E, z_h, labels = [r"$z_E$", r"$z_h$"], path = self.latent_pics, plotname = "zE_vs_zh", epo = epoch)
            self.kde_plot(z_E, z_h, labels = [r"$z_E$", r"$z_h$"], path = self.latent_pics, plotname = "kde_zE_vs_zh", epo = epoch)
        
        # 3)
        if z_h is not None:
            self.structural_plot(z_h, time_ind.squeeze().detach().cpu().numpy(), aux_name = "Time", path = self.latent_pics, plotname = "zh_with_time", epo = epoch)
        self.structural_plot(z_E, time_ind.squeeze().detach().cpu().numpy(), aux_name = "Time", path = self.latent_pics, plotname = "zE_with_time", epo = epoch)
       
        self.comparison_plot(z0, z_h_z0, labels = [r"$z_0$", r"$zhz0$"], path = self.latent_pics, plotname = "z0_vs_zhz0", epo = epoch)
        self.kde_plot(z0, z_h_z0, labels = [r"$z_0$", r"$zhz0$"], path = self.latent_pics, plotname = "kde_z0_vs_zhz0", epo = epoch)
        self.structural_plot(z0, time_ind.squeeze().detach().cpu().numpy(), aux_name = "Time", path = self.latent_pics, plotname = "z0_with_time", epo = epoch)
        self.structural_plot(z_h_z0, time_ind.squeeze().detach().cpu().numpy(), aux_name = "Time", path = self.latent_pics, plotname = "zhz0_with_time", epo = epoch)

    def vae_qualitative_analysis(self, 
                            prior_z, z_h, z_E,
                            
                            time_ind,
                            discrete = False,
                            epoch = ""):
        '''
        VAE
        '''
        self.logger.info(f"Computing plots (epoch = {epoch}) ... ")
        # Latent represenataions
        # Normalize z_h for comparison with the prior (mean 0 std 1)
        # if z_h is not None:
        #     z_h = (z_h - z_h.mean(dim = 0, keepdim = True)) / torch.clamp(z_h.std(dim = 0, keepdim = True), min=1e-6) 
        # z_E = (z_E - z_E.mean(dim = 0, keepdim = True)) / torch.clamp(z_E.std(dim = 0, keepdim = True), min=1e-6) 
        # PCA on z
        if z_h is not None:
            z_h = z_h.detach().cpu().numpy()
        z_E = z_E.detach().cpu().numpy()
        pca = PCA(n_components=2)
        if z_h is not None:
            pca.fit(np.concatenate((z_E, z_h), axis = 0))
            z_E, z_h = pca.transform(z_E), pca.transform(z_h)
        else:
            pca.fit(z_E)
            z_E= pca.transform(z_E)     
        # 1)
        if z_h is not None:
            self.comparison_plot(z_E, z_h, labels = [r"$p_{E}(z)$", "reparam"], path = self.latent_pics, plotname = "zE_vs_ze_gaus", epo = epoch)
        # 2)
        # self.comparison_plot(z_h, z_E, labels = [r"$p_{h}(z)$", r"$q_{E}(z|x)$"], path = self.latent_pics, plotname = "zh_vs_posteriorZ", epo = epoch)
        
        # 3)
        if z_h is not None:
            self.structural_plot(z_h, time_ind.squeeze().detach().cpu().numpy(), aux_name = "Time", path = self.latent_pics, plotname = "posteriorZ_on_Gaussian_with_time", epo = epoch)
        self.structural_plot(z_E, time_ind.squeeze().detach().cpu().numpy(), aux_name = "Time", path = self.latent_pics, plotname = "posteriorZ_with_time", epo = epoch)
        
    def rul_plot(self, pred, y, title = ""):
        fig = plt.figure(figsize=(12, 8), 
                            dpi = 600) 
        axes = fig.subplots()
        
        axes.plot(y[:], c = "blue", linewidth=2.5, alpha=1, label = "Ground Truth")
        axes.plot(pred[:], c = "red", linewidth=2.5, alpha=1.0, ls = "--", label = "Prediction")

        plt.xlabel("Time", fontsize=20, fontweight='bold')
        plt.ylabel("Remaining useful life", fontsize=20, fontweight='bold')
        plt.xticks(fontweight='bold', fontsize = 20)   
        plt.yticks(fontweight='bold', fontsize = 20)
        legend = axes.legend(fontsize=20, loc='upper right', bbox_to_anchor=(0.5, 1.1))
        plt.tight_layout()
        plt.savefig(os.path.join(self.data_pics, f"instance_{title}_rul.png" ))    
        plt.clf()   
        plt.close(fig)