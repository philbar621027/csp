import cupy as cp

def fsm1(InXog, alpha, nfft = 8192, nl = 128, start = 10): 
  ntest = nfft*nl
  x = InXog[start:start+ntest]

  xafp = cp.fft.fft(x * cp.exp(1j*cp.pi*alpha*cp.arange(ntest)))
  xafn = cp.fft.fft(x * cp.exp(1j*cp.pi*(-alpha)*cp.arange(ntest)))
  xaf = xafp * cp.conj(xafn)

  res = cp.reshape(xaf,(nfft,nl)) @ cp.hamming(nl)
  return res/(ntest)