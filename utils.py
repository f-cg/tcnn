def count_parameters(m, thresh=1e-4):
    r'''
    routers及leaves的参数:w,b,beta,Q.
    即 \sum_{i=1}^{L}((dim+1)*2^{i-1}+2^{i-1})+2^{L}*n_class
    '''
    paras = m.parameters()
    return sum(p[p.abs() > thresh].numel() for p in paras if p.requires_grad)


def check_on_gpu(model):
    for k, v in model.state_dict().items():
        print(k, v.device)
        if v.device == 'cpu':
            exit(-1)
