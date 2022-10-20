import constitutiveX as cx
import numpy as np

def jaumann_rotate_3d_python(L,sigma,del_t):
    W=0.5*(L-L.T)
    sigma_out = cx.mandel_to_tensor_3d(sigma)
    sigma_out += del_t*(sigma_out@W.T+W@sigma_out)
    return cx.tensor_to_mandel_3d(sigma_out)

def test_mandel_transformations():
    a = np.arange(9,dtype=np.float64).reshape(3,3)
    sym = 0.5*(a+a.T)
    mandel = np.array([sym[0,0],sym[1,1],sym[2,2],2**0.5*sym[1,2],2**0.5*sym[0,2],2**0.5*sym[0,1]])
    np.testing.assert_allclose(mandel,cx.tensor_to_mandel_3d(sym))
    np.testing.assert_allclose(sym, cx.mandel_to_tensor_3d(mandel))

def test_jaumann_rotate():
    L=np.random.random((3,3))
    sigma=np.random.random(6)
    L=np.arange(9,dtype=np.float64)
    sigma=np.arange(6,dtype=np.float64)
    del_t=0.01
    sigma_python = jaumann_rotate_3d_python(L.reshape(3,3),sigma,del_t)
    cx.jaumann_rotate_3d(L,sigma,del_t)
    np.testing.assert_allclose(sigma,sigma_python)

if __name__ == "__main__":
    test_mandel_transformations()
    test_jaumann_rotate()

