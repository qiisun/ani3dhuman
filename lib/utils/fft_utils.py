import torch
import torch.fft as fft


def get_low_or_high_fft(x, scale, is_low=True):
    # note that fft do not support bfloat16 
    # FFT
    x = x.to(torch.float32)
    x_freq = fft.fftn(x, dim=(-2, -1))
    x_freq = fft.fftshift(x_freq, dim=(-2, -1))
    B, C, T, H, W = x_freq.shape
    
    # extract
    if is_low:
        mask = torch.zeros((B, C, T, H, W), device=x.device)
        crow, ccol = H // 2, W // 2
        mask[..., crow - int(crow * scale):crow + int(crow * scale), ccol - int(ccol * scale):ccol + int(ccol * scale)] = 1
    else:
        mask = torch.ones((B, C, T, H, W), device=x.device)
        crow, ccol = H // 2, W //2
        mask[..., crow - int(crow * scale):crow + int(crow * scale), ccol - int(ccol * scale):ccol + int(ccol * scale)] = 0
    x_freq = x_freq * mask
    
    # IFFT
    x_freq = fft.ifftshift(x_freq, dim=(-2, -1))
    x_filtered = fft.ifftn(x_freq, dim=(-2, -1)).real
    x_filtered = x_filtered.to(torch.bfloat16)
    return x_filtered

if __name__ == '__main__':
    x = torch.randn((2, 16, 20, 64, 64)).cuda()
    x_high = get_low_or_high_fft(x, 4, is_low=False)
    print(x_high.shape)