import numpy as np

def left_shift(x):
    dims = x.shape
    # print(dims)
    x = np.pad(x, [(0,0), (0,0), (0,0), (1,0)])
    x = x.reshape(dims[0], dims[1], dims[3]+1, dims[2])
    x = x[:, :, 1:, :]
    # print(x.shape)
    x = x.reshape(*dims)
    return x

def right_shift(x):
    dims = x.shape
    x = np.pad(x, [(0,0), (0,0), (0,0), (0,1)])
    x = x.reshape(dims[0], dims[1], dims[3]+1, dims[2])
    x = x[:, :, :-1, :]
    x = x.reshape(*dims)
    return x

def generate_samples(qlen, klen, d_model):
    batch = 5
    head = 8
    q = np.random.randn(batch, head, qlen, d_model)
    U = np.random.randn(batch, head, klen, d_model)
    # U = np.transpose(U, (0, 1, 3, 2))
    offset = klen-qlen # if klen>qlen, start qlen at offset position
    M = np.zeros((batch, head, qlen, klen))
    for b in range(batch):
        for h in range(head):
            for i in range(qlen):
                for j in range(klen):
            
                    M[b, h, i, j] = q[b, h, i] @ U[b, h, abs(i-j+offset)] # this is a vector dot product
    return M, q, U

def rel_position(q, U):
    
        m, n = q.shape[2], U.shape[2]        
        lower_mask, upper_mask = np.tril(np.ones((5, 8, n,n)))[:,:, n-m:], np.triu(np.ones((5, 8, n,n)), k=1)[:, :, n-m:]
        
        # print(q.shape)
        # print(np.transpose(np.flipud(U), (0, 1, 3, 2)).shape)
        lower_diag = left_shift(q@np.transpose(np.flipud(U), (0, 1, 3, 2)))

        if m < n:
            
            upper_diag = right_shift(q@np.transpose(U[:, :, :m-n], (0, 1, 3, 2)))
            upper_diag = np.pad(upper_diag, [(0,0), (0,0), (0,0), (n-m,0)])
        else:
            
            upper_diag = right_shift(q@np.transpose(U, (0, 1, 3, 2)))
            
        return upper_diag*upper_mask + lower_diag*lower_mask # these zero out the garbage parts
    

qlen = 3
klen = 5
d_model = 12
M, q, U = generate_samples(qlen, klen, d_model)
pred = rel_position(q, U)
print("Correct answer:\n", np.round(M, 2))
print("Shifted algorithm:\n", np.round(pred, 2))
print("Match?", np.allclose(M, pred))