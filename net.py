import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import initialize_weights

class MIL(nn.Module):
    def __init__(self,img_sz = (28,28),data_type = "matrix", model_type = "nn", attention_flag = False, gate = False):
        super(MIL, self).__init__()
        self.L = 1
        self.D = 256
        self.K = 1
        self.data_type = data_type
        self.model_type = model_type
        self.attention_flag = attention_flag
        self.r1 = 2
        self.r2 = 2
        self.r3 = 2
        self.d1,self.d2 = img_sz[0],img_sz[1]
        self.vector_dim = self.d1 * self.d2
        self.feat_map_sz1,self.feat_map_sz2 = self.d1//4,self.d2//4

        self.gate = gate
        self.n_feat = 50 * self.feat_map_sz1 * self.feat_map_sz2
        if self.model_type != "vector_data":
            self.feature_extractor_part1 = nn.Sequential(
                        # nn.Conv2d(3, 20, kernel_size=5, padding = 2),
                        nn.Conv2d(1, 20, kernel_size=5, padding = 2),
                        nn.BatchNorm2d(20),
                        nn.ReLU(),
                        nn.MaxPool2d(2, stride=2),
                        nn.Conv2d(20, 50, kernel_size=5, padding = 2),
                        nn.BatchNorm2d(50),
                        nn.ReLU(),
                        nn.MaxPool2d(2, stride=2)
                    )
            # P x 50 x 4 x 4
            self.feature_extractor_part2 = nn.Sequential(
                        nn.Linear(self.n_feat, self.L, bias = False),
                        # nn.ReLU(),
                        ## add total variation after sigmoid
                        # nn.Sigmoid()
                    )
        class Attn_Net(nn.Module):
            def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
                super(Attn_Net, self).__init__()
                self.module = [
                    nn.Linear(L, D),
                    nn.Tanh()]

                if dropout:
                    self.module.append(nn.Dropout(0.25))

                self.module.append(nn.Linear(D, n_classes))
                
                self.module = nn.Sequential(*self.module)
            
            def forward(self, x):
                return self.module(x), x # N x n_classes

        class Attn_Net_Gated(nn.Module):
            def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
                super(Attn_Net_Gated, self).__init__()
                self.attention_a = [
                    nn.Linear(L, D),
                    nn.Tanh()]
                
                self.attention_b = [nn.Linear(L, D),
                                    nn.Sigmoid()]
                if dropout:
                    self.attention_a.append(nn.Dropout(0.25))
                    self.attention_b.append(nn.Dropout(0.25))

                self.attention_a = nn.Sequential(*self.attention_a)
                self.attention_b = nn.Sequential(*self.attention_b)
                
                self.attention_c = nn.Linear(D, n_classes)

            def forward(self, x):
                a = self.attention_a(x)
                b = self.attention_b(x)
                A = a.mul(b)
                A = self.attention_c(A)  # N x n_classes
                return A, x
        if self.model_type == "nn":
            if self.data_type == "tensor":
                '''
                ADNI:96x120x96;
                Patch size: 16x20x16; 
                the number: 6x6x6 = 216
                Therefore, the input tensor is 16x20x16
            ''' 
                self.feature_extractor_part1 = nn.Sequential(
                        nn.Conv3d(1,20,5,stride = 1,padding = 0),
                        ## output size: 12x16x12
                        nn.BatchNorm3d(20),
                        nn.ReLU(inplace=True),
                        nn.MaxPool3d(2, 2),
                        ## output size : 6x8x6
                        nn.Conv3d(20, 50, 3, stride = 1, padding=0),
                        ## output size: 4 x 6 x 4; note here change kernel size from 5 to 3, since the the input size is small
                        nn.BatchNorm3d(50),
                        nn.ReLU(inplace=True),
                        nn.MaxPool3d(2, 2),
                        ## output size: 2x3x2
                    )
                self.feature_extractor_part2 = nn.Sequential(
                    nn.Linear(50 * 2 * 3 * 2, self.L),
                    # nn.ReLU(),
                    # nn.Sigmoid()
                )
        elif self.model_type == "vector_data":
            '''
            CAMELYON 16, input format: vector
            '''
            ## self.feature_extrator_part 1 is omit
            ## two-stage algorithms
            hidden_size = 512
            atten_hidden_size = 256
            dropout = False
            # gate = False
            if self.gate:
                attention_net = Attn_Net_Gated(L=hidden_size, D=atten_hidden_size, dropout=dropout, n_classes=1)
            else:
                attention_net = Attn_Net(
                    L=hidden_size, D=atten_hidden_size, dropout=dropout, n_classes=1)


            fc = [nn.Linear(self.vector_dim, hidden_size), nn.ReLU()]
            if self.attention_flag == "attention":
                # fc.append(attention_net)
                self.attention_net = Attn_Net(
                    L=hidden_size, D=atten_hidden_size, dropout=dropout, n_classes=1)
            # elif self.attention_flag == "mean":
                

            self.feature_extractor = nn.Sequential(*fc)
            self.classifier = nn.Sequential(
                nn.Linear(hidden_size,1)
                )
        elif self.model_type == "attention":
            dropout = False
            hidden_size = 512  # (512,256)
            atten_hidden_size = 256
            if self.gate:
                self.attention_net = Attn_Net_Gated(L=hidden_size, D=atten_hidden_size, dropout=dropout, n_classes=1)
            else:
                self.attention_net = Attn_Net(
                    L=hidden_size, D=atten_hidden_size, dropout=dropout, n_classes=1)

            fc = [nn.Linear(self.n_feat, hidden_size), nn.ReLU()]
            # fc.append(attention_net)

            self.feature_extractor_part2 = nn.Sequential(*fc)

            self.classifier = nn.Sequential(
                nn.Linear(hidden_size,1),
                nn.Sigmoid()
                )
        elif self.model_type == "mean":
                hidden_size = 512
                fc = [nn.Linear(self.n_feat, hidden_size), nn.ReLU()]
                self.feature_extractor_part2 = nn.Sequential(*fc)
                self.classifier = nn.Sequential(
                nn.Linear(hidden_size,1),
                nn.Sigmoid()
                )
        else:
            ### linear form 
            self.feature_extractor_part2 = nn.Sequential(
                nn.Linear(self.d1 * self.d2, 1, bias = False)
                )
        # initialize_weights(self)


    def tt(self,x):
        x = torch.tensordot(x,self.g1,dims = ([2],[0]))
        # x = torch.tanh(x)
        # print("x1 dim: ",x.shape)
        x = torch.tensordot(x,self.g2,dims = ([1,3],[1,0]))
        # print("x2 dim: ",x.shape)
        # x = torch.tanh(x)
        x = torch.tensordot(x,self.g3,dims = ([1,2],[1,0]))
        # x = torch.tanh(x)
        # print("x3 dim: ",x.shape)
        return x

    def cam(self,x):
        x = torch.sum(x, axis = (2,3))
        # x =  torch.sum(x * self.W, axis = 1)
        x = self.cam_conv(x)
        return x
    def forward(self, x):
        x = x.squeeze(0)
        if self.model_type == "nn":
            H = self.feature_extractor_part1(x)
            # print("H.shape: ",H.shape)
            if self.type == "tensor":
                H = H.view(-1, 50 * 2 * 3 * 2)
            else:
                H = H.reshape(-1, self.n_feat)
                pass
            Y_hat = self.feature_extractor_part2(H)  # NxL, L = 1

            # print("H shape: ",H.shape)
            ####--------------------------TT -train--------
            # Y_hat = self.tt(H).reshape(-1,1)

            #####------------------------ CAM--------------------
            # H = self.cam_conv(H.view(-1,4 * 4))
            # H = H.view(-1, 50)
            # Y_hat = self.cam(H).reshape(-1,1)
            return Y_hat,H.detach().cpu().numpy()

        elif self.model_type == "attention":
            H = self.feature_extractor_part1(x)
            H = H.reshape(-1, self.n_feat)
            H = self.feature_extractor_part2(H)  # NxL

            ## inject attentino module
            A,H = self.attention_net(H)  # NxK
            A = torch.transpose(A, 1, 0)  # KxN
            A = F.softmax(A, dim=1)  # softmax over N

            M = torch.mm(A, H)  # KxL

            Y_prob = self.classifier(M)
            # Y_hat = torch.ge(Y_prob, 0.5).float()

            return Y_prob,A
        elif self.model_type == "mean":
            H = self.feature_extractor_part1(x)
            H = H.reshape(-1, self.n_feat)
            H = self.feature_extractor_part2(H)  # NxL
            N = H.shape[0]
            # print("H shape: ",H.shape)
            if N != 0 :
                A = torch.tensor([1/N]*N).unsqueeze(0).type_as(H)
            else:
                A = torch.tensor([0]*N).unsqueeze(0).type_as(H)
            A = F.softmax(A, dim=1)  # softmax over N
            M = torch.mm(A, H)  # KxL
            Y_prob = self.classifier(M)
            return Y_prob,A
        ### ----------- CAMEYON 16
        elif self.model_type == "vector_data":
            # print("x shape: ",x.shape)
            # H = self.feature_extractor_part1(x.view(-1, self.vector_dim))
            ### attention
            # A, H = self.feature_extractor(x)
            # A = torch.transpose(A, 1, 0)
            # A = F.softmax(A , dim=1)
            # M = H  * A.t()
            ## mean-pooling
            H = self.feature_extractor(x)
            M = H
            A = 0
            if self.attention_flag == "attention":
                ## inject attentino module
                A,H = self.attention_net(H)  # NxK
                A = torch.transpose(A, 1, 0)  # KxN
                A = F.softmax(A, dim=1)  # softmax over N

                M = torch.mm(A, H)  # KxL
            elif self.attention_flag == "mean":
                N = H.shape[0]
                A = torch.tensor([1/N]*N).unsqueeze(0).type_as(H)
                A = F.softmax(A, dim=1)  # softmax over N
                M = torch.mm(A, H)  # KxL
            ## use additive 
            # M = A @ H

            ### A: 1 x p
            ### H: p x hidden_size
            
            patch_logits = self.classifier(M) ## return a p x 1 vector
            # logits = torch.sum(patch_logits, dim=0, keepdim=True)
            # logits /= 20  ## 800
            if self.attention_flag is not None:
                return patch_logits,A
            else:
                return patch_logits,0

        ### --------------- linear transformation
        else:
            H = self.feature_extractor_part2(x.view(-1,self.d1  * self.d2))
            return H,1
    # AUXILIARY METHODS
    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        _, Y_hat, _ = self.forward(X)
        error = 1. - Y_hat.eq(Y).cpu().float().mean().data.item()

        return error, Y_hat

    def calculate_objective(self, X, Y):
        Y = Y.float()
        Y_prob, _, A = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli

        return neg_log_likelihood, A