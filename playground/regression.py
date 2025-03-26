import torch


def closed_form(q, k, v, KK, KV, eps=1e-6):
    if KK is None:
        KK = k.t() @ k + eps * torch.eye(k.size(1)).cuda()
        KV = torch.zeros(k.size(1), v.size(1)).cuda()
    else:
        KK = KK + k.t() @ k

    KV = KV + v @ k.t()
    return KV @ torch.inverse(KK) @ q, KK, KV


def main():
    seq_len = 512
    dim = 128
    Q = torch.randn(seq_len, dim).cuda()
    K = torch.randn(seq_len, dim).cuda()
    V = torch.randn(seq_len, dim).cuda()
    K = K / torch.norm(K, dim=-1, keepdim=True)
    KK = None
    KV = None
    for t in range(seq_len):
        q = Q[t]
        k = K[t].unsqueeze(0)
        v = V[t].unsqueeze(0)
        out, KK, KV = closed_form(q, k, v, KK, KV)
        # print the norm of output
        print(torch.norm(out))


if __name__ == '__main__':
    main()