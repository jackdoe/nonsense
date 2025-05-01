```
def _unitary(self, h_real):
    H = 0.5*(A + A^T)                # make a symmetric real matrix
    U = expm(−i·H·dt)               # a learnable unitary via matrix exponential
    return h_c + (h_c @ U)
h_c = self._unitary(self.ln2(h_c.real).to(torch.cfloat))
```

experimenting with unitary feature mixer `U = e^{−iHΔt}` 


