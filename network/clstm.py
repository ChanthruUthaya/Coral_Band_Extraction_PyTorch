import torch
import torch.nn as nn
from torch.autograd import Variable



if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


class ConvLSTMCell(nn.Module):


    def __init__(self, in_channels, hidden_channels, kernel_size, bias):

        super(ConvLSTMCell, self).__init__()

        self.input_dim = in_channels
        self.hidden_dim = hidden_channels

        self.kernel_size = kernel_size
        self.padding = kernel_size[0]//2, kernel_size[1]//2
        self.bias = bias

        self.conv = nn.Conv2d( in_channels=self.input_dim+self.hidden_dim,
                            out_channels=4*self.hidden_dim,
                            kernel_size = self.kernel_size,
                            padding = self.padding,    
                            bias= self.bias
                                )
        
    def forward(self, input_tensor, current_state):

        h_state, c_state = current_state

        combinded = torch.cat([input_tensor, h_state], dim=1)

        combinded_conv = self.conv(combinded)

        cc_i, cc_f, cc_o, cc_g = torch.split(combinded_conv, self.hidden_dim, dim=1)

        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.sigmoid(cc_g)

        c_next = f * c_state + i*g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next
        
    def init_hidden(self, b, h, w):
        return (torch.zeros(b, self.hidden_dim, h, w).to(DEVICE),
                torch.zeros(b, self.hidden_dim, h, w).to(DEVICE))


class ConvLSTM(nn.Module):

    def __init__(self, in_channels, hidden_channels, kernel_size, batch_first=True, bias=True):

            super(ConvLSTM, self).__init__()

            self.input_dim = in_channels
            self.hidden_dim = hidden_channels
            self.kernel_size = kernel_size
            self.batch_first = batch_first
            self.bias = bias

            self.clstm = ConvLSTMCell(in_channels = self.input_dim,
                                              hidden_channels = self.hidden_dim,
                                              kernel_size = self.kernel_size,
                                              bias=self.bias)
                                            

    def forward(self, input_tensor):

        b, seq_len, _, h, w = input_tensor.size()

        h, c = self.clstm.init_hidden(b, h, w)
 
        cell_output = []

        for t in range(seq_len):
            h, c = self.clstm(input_tensor[:,t,:,:,:], [h,c])

            cell_output.append(h)

        output = torch.stack(cell_output, dim = 1)
       

        return output, (h,c)



class ConvBLSTM(nn.Module):

    def __init__(self, in_channels, hidden_channels, kernel_size, batch_first = True):
        super(ConvBLSTM, self).__init__()

        self.forward_net = ConvLSTM(in_channels, hidden_channels//2, kernel_size, batch_first = batch_first)
        self.backward_net = ConvLSTM(in_channels, hidden_channels//2, kernel_size, batch_first = batch_first)
    
    def forward(self, input_forward, input_backward):

        out_f, _ = self.forward_net(input_forward)
        out_b, _ = self.backward_net(input_backward)

        reversed_idx = list(reversed(range(out_b.shape[1])))
        out_b = out_b[:, reversed_idx, ...] # reverse temporal outputs.
        ycat = torch.cat((out_f, out_b), dim=2)

        return ycat


if __name__ == "__main__":

    x1 = torch.randn([3, 1024, 16, 16]).to(DEVICE)
    x2 = torch.randn([3, 1024, 16, 16]).to(DEVICE)
    x3 = torch.randn([3, 1024, 16, 16]).to(DEVICE)

    cblstm = ConvBLSTM(in_channels=1024, hidden_channels=2048, kernel_size=(3, 3), batch_first=True).to(DEVICE)

    x_fwd = torch.stack([x1, x2, x3], dim=1)
    x_rev = torch.stack([x3, x2, x1], dim=1)

    out = cblstm(x_fwd, x_rev)
    print (out.shape)
    out.sum().backward()

