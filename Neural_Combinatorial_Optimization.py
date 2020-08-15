import math
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader


from IPython.display import clear_output
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import LambdaLR

USE_CUDA = False

class TSPDataset(Dataset):
    
    def __init__(self, num_nodes, num_samples, random_seed=-1):
        super(TSPDataset, self).__init__()
        if random_seed != -1:
            torch.manual_seed(random_seed)

        self.data_set = []
        for l in tqdm(range(num_samples)):
            x = torch.FloatTensor(2, num_nodes).uniform_(0, 1)
            self.data_set.append(x)

        self.size = len(self.data_set)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data_set[idx]
    

def reward(sample_solution, USE_CUDA=False):
    """
    Args:
        sample_solution seq_len of [batch_size]
    """
    batch_size = sample_solution[0].size(0)
    n = len(sample_solution)
    tour_len = Variable(torch.zeros([batch_size]))
    
    if USE_CUDA:
        tour_len = tour_len.cuda()

    for i in range(n - 1):
        tour_len += torch.norm(sample_solution[i] - sample_solution[i + 1], dim=1)
    
    tour_len += torch.norm(sample_solution[n - 1] - sample_solution[0], dim=1)

    return tour_len


class Attention(nn.Module):
    def __init__(self, hidden_size, name='Bahdanau', use_cuda=USE_CUDA):
        super(Attention, self).__init__()
        
        self.name = name
        
        if name == 'Bahdanau':
            self.W_query = nn.Linear(hidden_size, hidden_size)
            self.W_ref   = nn.Conv1d(hidden_size, hidden_size, 1, 1)

            V = torch.FloatTensor(hidden_size)
            if use_cuda:
                V = V.cuda()  
            self.V = nn.Parameter(V)
            self.V.data.uniform_(-(1. / math.sqrt(hidden_size)) , 1. / math.sqrt(hidden_size))

    def forward(self, query, ref):
        """
        Args: 
            query: [batch_size x hidden_size]
            ref:   [batch_size x seq_len x hidden_size]
        """
        
        batch_size = ref.size(0)
        seq_len    = ref.size(1)
        
        if self.name == 'Bahdanau':
            ref = ref.permute(0, 2, 1)
            query = self.W_query(query).unsqueeze(2)  # [batch_size x hidden_size x 1]
            ref   = self.W_ref(ref)  # [batch_size x hidden_size x seq_len] 
            expanded_query = query.repeat(1, 1, seq_len) # [batch_size x hidden_size x seq_len]
            V = self.V.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1) # [batch_size x 1 x hidden_size]
            logits = torch.bmm(V, torch.tanh(expanded_query + ref)).squeeze(1)
            
        elif self.name == 'Dot':
            query  = query.unsqueeze(2)
            logits = torch.bmm(ref, query).squeeze(2) #[batch_size x seq_len x 1]
            ref = ref.permute(0, 2, 1)
        
        else:
            raise NotImplementedError
        
 
        return ref, logits

class GraphEmbedding(nn.Module):
    def __init__(self, input_size, embedding_size, use_cuda=USE_CUDA):
        super(GraphEmbedding, self).__init__()
        self.embedding_size = embedding_size
        self.use_cuda = use_cuda
        
        self.embedding = nn.Parameter(torch.FloatTensor(input_size, embedding_size)) 
        self.embedding.data.uniform_(-(1. / math.sqrt(embedding_size)), 1. / math.sqrt(embedding_size))
        
    def forward(self, inputs):
        batch_size = inputs.size(0)
        seq_len    = inputs.size(2)
        embedding = self.embedding.repeat(batch_size, 1, 1)  
        embedded = []
        inputs = inputs.unsqueeze(1)
        for i in range(seq_len):
            embedded.append(torch.bmm(inputs[:, :, :, i].float(), embedding))
        embedded = torch.cat(embedded, 1)
        return embedded

class PointerNet(nn.Module):
    def __init__(self, 
            embedding_size,
            hidden_size,
            seq_len,
            n_glimpses,
            attention,
            C=10,
            use_tanh=False,
            use_cuda=USE_CUDA):
        super(PointerNet, self).__init__()
        
        self.embedding_size = embedding_size
        self.hidden_size    = hidden_size
        self.n_glimpses     = n_glimpses
        self.seq_len        = seq_len
        self.use_cuda       = use_cuda
        self.use_tanh       = use_tanh
        self.C              = C
        
        
        self.embedding = GraphEmbedding(2, embedding_size, use_cuda=use_cuda)
        self.encoder = nn.LSTM(embedding_size, hidden_size, batch_first=True)
        self.decoder = nn.LSTM(embedding_size, hidden_size, batch_first=True)
        self.pointer = Attention(hidden_size, name=attention, use_cuda=use_cuda)
        self.glimpse = Attention(hidden_size, name=attention, use_cuda=use_cuda)
        
        self.decoder_start_input = nn.Parameter(torch.FloatTensor(embedding_size))
        self.decoder_start_input.data.uniform_(-(1. / math.sqrt(embedding_size)), 1. / math.sqrt(embedding_size))
        
    def apply_mask_to_logits(self, logits, mask, idxs): 
        batch_size = logits.size(0)
        clone_mask = mask.clone()

        if idxs is not None:
            clone_mask[[i for i in range(batch_size)], idxs.data] = 1
            logits[clone_mask] = -np.inf
        return logits, clone_mask

            
    def forward(self, inputs):
        """
        Args: 
            inputs: [batch_size x 2 x seq_len]
        """
        batch_size = inputs.size(0)
        seq_len    = inputs.size(2)
        assert seq_len == self.seq_len
        
        embedded = self.embedding(inputs)
        encoder_outputs, (hidden, context) = self.encoder(embedded)
        
        
        prev_probs = []
        prev_idxs = []
        mask = torch.zeros(batch_size, seq_len).bool()
        if self.use_cuda:
            mask = mask.cuda()
            
        idxs = None
       
        decoder_input = self.decoder_start_input.unsqueeze(0).repeat(batch_size, 1)
        
        for i in range(seq_len):
            
            
            _, (hidden, context) = self.decoder(decoder_input.unsqueeze(1), (hidden, context))
            
            query = hidden.squeeze(0)
            for i in range(self.n_glimpses):
                ref, logits = self.glimpse(query, encoder_outputs)
                logits, mask = self.apply_mask_to_logits(logits, mask, idxs)
                query = torch.bmm(ref, F.softmax(logits, dim=1).unsqueeze(2)).squeeze(2) 
                
                
            _,logits = self.pointer(query, encoder_outputs)
            
            if self.use_tanh:
                logits = self.C*torch.tanh(logits)
            else:
                logits = logits/2.0

            logits, mask = self.apply_mask_to_logits(logits, mask, idxs)
            probs = F.softmax(logits, dim=1)
 
            idxs = probs.multinomial(1).squeeze(1)
            for old_idxs in prev_idxs:
                if old_idxs.eq(idxs).data.any():
                    print(seq_len)
                    print(' RESAMPLE!')
                    idxs = probs.multinomial(1).squeeze(1)
                    break
            decoder_input = embedded[[i for i in range(batch_size)], idxs.data, :] 
            
            prev_probs.append(probs)
            prev_idxs.append(idxs)
            
        return prev_probs, prev_idxs
    
    def greedy(self, inputs):
        """
        Args: 
            inputs: [batch_size x 2 x seq_len]
        """
        batch_size = inputs.size(0)
        seq_len    = inputs.size(2)
        assert seq_len == self.seq_len
        
        embedded = self.embedding(inputs) #embedded : [batch_size, seq_len, embedding_size]
        encoder_outputs, (hidden, context) = self.encoder(embedded)
        
        
        prev_probs = []
        prev_idxs = []
        mask = torch.zeros(batch_size, seq_len).bool()
        if self.use_cuda:
            mask = mask.cuda()
            
        idxs = None
       
        decoder_input = self.decoder_start_input.unsqueeze(0).repeat(batch_size, 1)
        
        for i in range(seq_len):
            
            
            _, (hidden, context) = self.decoder(decoder_input.unsqueeze(1), (hidden, context))
            
            query = hidden.squeeze(0)
            for i in range(self.n_glimpses):
                ref, logits = self.glimpse(query, encoder_outputs)
                #logits = self.C*torch.tanh(logits)
                logits, mask = self.apply_mask_to_logits(logits, mask, idxs)
                query = torch.bmm(ref, F.softmax(logits, dim=1).unsqueeze(2)).squeeze(2) 
                
                
            _,logits = self.pointer(query, encoder_outputs)
            
            if self.use_tanh:
                logits = self.C*torch.tanh(logits)
            else:
                logits = logits/2.0

            logits, mask = self.apply_mask_to_logits(logits, mask, idxs)
            probs = F.softmax(logits, dim=1)
            #idxs = probs.multinomial(1).squeeze(1)
            idxs = probs.argmax(dim=1)

            for old_idxs in prev_idxs:
                if old_idxs.eq(idxs).data.any():
                    print(seq_len)
                    print(' RESAMPLE!')
                    idxs = probs.multinomial(1).squeeze(1)
                    break
            
            decoder_input = embedded[[i for i in range(batch_size)], idxs.data, :] 
            
            prev_probs.append(probs)
            prev_idxs.append(idxs)
            
        return prev_probs, prev_idxs

class CombinatorialRL(nn.Module):
    def __init__(self, 
            embedding_size,
            hidden_size,
            seq_len,
            n_glimpses,
            tanh_exploration,
            use_tanh,
            reward,
            attention,
            use_cuda=USE_CUDA):
        super(CombinatorialRL, self).__init__()
        self.reward = reward
        self.use_cuda = use_cuda
        
        self.actor = PointerNet(
                embedding_size,
                hidden_size,
                seq_len,
                n_glimpses,
                attention,
                C=tanh_exploration,
                use_tanh = use_tanh,
                use_cuda = use_cuda)


    def forward(self, inputs):
        """
        Args:
            inputs: [batch_size, input_size, seq_len]
        """
        batch_size = inputs.size(0)
        #input_size = inputs.size(1)
        #seq_len    = inputs.size(2)
        
        probs, action_idxs = self.actor(inputs)
        #probs : [seq_len, batch_size, seq_len]
        #action_idxs : [seq_len, batch_size]
        actions = []
        
        inputs = inputs.transpose(1, 2)
        for action_id in action_idxs:
            actions.append(inputs[[x for x in range(batch_size)], action_id.data, :])

            
        action_probs = []    
        for prob, action_id in zip(probs, action_idxs):
            action_probs.append(prob[[x for x in range(batch_size)], action_id.data])

        R = self.reward(actions, self.use_cuda)
        
        return R, action_probs, actions, action_idxs

    def greedy(self, inputs):
        """
        Args:
            inputs: [batch_size, input_size, seq_len]
        """
        batch_size = inputs.size(0)
        
        probs, action_idxs = self.actor.greedy(inputs) 
        #probs : [seq_len, batch_size, seq_len]
        #action_idxs : [seq_len, batch_size]
        actions = []
        
        inputs = inputs.transpose(1, 2)
        for action_id in action_idxs:
            actions.append(inputs[[x for x in range(batch_size)], action_id.data, :])

            
        action_probs = []    
        for prob, action_id in zip(probs, action_idxs):
            action_probs.append(prob[[x for x in range(batch_size)], action_id.data])

        R = self.reward(actions, self.use_cuda)
        
        return R, action_probs, actions, action_idxs

class CriticNet(nn.Module):
    def __init__(self, 
            embedding_size,
            hidden_size,
            seq_len,
            n_glimpses,
            attention,
            C=10,
            use_tanh=False,
            use_cuda=USE_CUDA,
            d=128):
        super(CriticNet, self).__init__()
        
        self.embedding_size = embedding_size
        self.hidden_size    = hidden_size
        self.n_glimpses     = n_glimpses
        self.seq_len        = seq_len
        self.use_cuda       = use_cuda
        self.use_tanh       = use_tanh
        self.C              = C
        
        
        self.embedding = GraphEmbedding(2, embedding_size, use_cuda=use_cuda)
        self.encoder = nn.LSTM(embedding_size, hidden_size, batch_first=True)
        self.process = nn.LSTM(embedding_size, hidden_size, batch_first=True)
        self.glimpse = Attention(hidden_size, name=attention, use_cuda=use_cuda)
        self.decoder1 = nn.Linear(hidden_size, d)
        self.relu = nn.ReLU()
        self.decoder2 = nn.Linear(d,1)
        
        self.process_start_input = nn.Parameter(torch.FloatTensor(embedding_size))
        self.process_start_input.data.uniform_(-(1. / math.sqrt(embedding_size)), 1. / math.sqrt(embedding_size))
            
    def forward(self, inputs):
        """
        Args: 
            inputs: [batch_size x 2 x seq_len]
        """
        batch_size = inputs.size(0)
        seq_len    = inputs.size(2)
        assert seq_len == self.seq_len
        
        #LSTM encoder
        embedded = self.embedding(inputs) #embedded : [batch_size, seq_len, embedding_size]
        encoder_outputs, (hidden, context) = self.encoder(embedded)

        #LSTM process block        
        process_input = self.process_start_input.unsqueeze(0).repeat(batch_size, 1)
        for i in range(3):
            
            
            _, (hidden, context) = self.process(process_input.unsqueeze(1), (hidden, context))
            
            query = hidden.squeeze(0)
            for i in range(self.n_glimpses):
                ref, logits = self.glimpse(query, encoder_outputs)
                query = torch.bmm(ref, F.softmax(logits, dim=1).unsqueeze(2)).squeeze(2)             
            
            process_input = query
            
            
        
        #decode last hidden into a scalar
        d_layer = self.decoder1(process_input)
        d_layer = self.relu(d_layer)
        baseline_pred = self.decoder2(d_layer)
        
        return baseline_pred

class TrainModel:
    def __init__(self, model, critic, val_dataset, 
                 batch_size=128, threshold=None, max_grad_norm=2., learningRate = 1e-3):
        self.model = model
        self.critic = critic
        
        self.val_dataset   = val_dataset
        self.batch_size = batch_size
        self.threshold = threshold
        
        self.val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        self.actor_optim   = optim.Adam(model.actor.parameters(), lr=learningRate)
        self.critic_optim = optim.Adam(critic.parameters(), lr=learningRate)
        self.max_grad_norm = max_grad_norm
        
        self.train_tour = []
        self.val_tour   = []
        
        self.epochs = 0
    
    def train_and_validate(self, rand_seeds,seq_length):
        lr_lambda = lambda epoch: 0.96 ** epoch
        scheduler_actor = LambdaLR(self.actor_optim,lr_lambda=lr_lambda)
        
        for seed in rand_seeds:
            train_dataset = TSPDataset(seq_length, 5000*self.batch_size,random_seed = seed)
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, 
                                      shuffle=True, num_workers=0)
            
            for batch_id, sample_batch in enumerate(train_loader):
                self.model.train()
                self.critic.train()

                inputs = Variable(sample_batch)
                if USE_CUDA: 
                    inputs = inputs.cuda()

                R, probs, actions, actions_idxs = self.model(inputs)
                baseline = self.critic(inputs)



                advantage = R - baseline

                logprobs = 0
                for prob in probs: 
                    logprob = torch.log(prob)
                    logprobs += logprob
                logprobs[logprobs < -1000] = 0.  

                reinforce = advantage * logprobs
                actor_loss = reinforce.mean()
                
                square_diff = (baseline-R)**2
                critic_loss = square_diff.mean()

                self.actor_optim.zero_grad()
                actor_loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(self.model.actor.parameters(),
                                    float(self.max_grad_norm), norm_type=2)

                self.actor_optim.step()

                self.critic_optim.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(),
                                    float(self.max_grad_norm), norm_type=2)
                self.critic_optim.step()
                
                self.train_tour.append(R.mean().item())

                
                if batch_id % 1000 == 0:    

                    self.model.eval()
                    for val_batch in self.val_loader:
                        inputs = Variable(val_batch)
                        if USE_CUDA: 
                            inputs = inputs.cuda()

                        R, probs, actions, actions_idxs = self.model(inputs)
                        self.val_tour.append(R.mean().item())
                
                if batch_id % 50 == 0:
                    self.plot(self.epochs)

            if self.threshold and self.train_tour[-1] < self.threshold:
                print("EARLY STOPPAGE!")
                break
            
            self.plot(self.epochs)
            
            torch.save({
                    'epoch' : self.epochs,
                    'model_state_dict' : self.model.state_dict(),
                    'optimizer_state_dict': self.actor_optim.state_dict(),
                    'critic_state_dict' : self.critic.state_dict(),
                    'critic_optim_state_dict': self.critic_optim.state_dict()
                    }, 'model_' + str(self.epochs) + '.tar')
            
            
            scheduler_actor.step()
                
            self.epochs += 1
                
    def plot(self, epoch):
        clear_output(True)
        plt.figure(figsize=(20,5))
        plt.subplot(131)
        plt.title('train tour length: epoch %s reward %s' % (epoch, self.train_tour[-1] if len(self.train_tour) else 'collecting'))
        plt.plot(self.train_tour)
        plt.grid()
        plt.subplot(132)
        plt.title('val tour length: epoch %s reward %s' % (epoch, self.val_tour[-1] if len(self.val_tour) else 'collecting'))
        plt.plot(self.val_tour)
        plt.grid()
        plt.savefig('plot' + str(epoch) + '.png')
        plt.show()
        plt.close('all')

val_size = 10000
embedding_size = 128
hidden_size    = 128
seq_length = 20
n_glimpses = 1
tanh_exploration = 10
use_tanh = True
attention = "Bahdanau"

beta = 0.9
max_grad_norm = 2.

val_20_dataset   = TSPDataset(20, val_size)





tsp_20_model = CombinatorialRL(
        embedding_size,
        hidden_size,
        seq_length,
        n_glimpses, 
        tanh_exploration,
        use_tanh,
        reward,
        attention=attention,
        use_cuda=USE_CUDA)


if USE_CUDA:
    tsp_20_model = tsp_20_model.cuda()
    
    
critic = CriticNet(
        embedding_size,
        hidden_size,
        seq_length,
        n_glimpses, 
        attention,
        C=tanh_exploration,
        use_tanh=use_tanh,
        use_cuda=USE_CUDA,
        d=hidden_size)

tsp_20_train = TrainModel(tsp_20_model,
                          critic, 
                        val_20_dataset)

seeds = [1,2,3,4,5]
tsp_20_train.train_and_validate(seeds,seq_length)










