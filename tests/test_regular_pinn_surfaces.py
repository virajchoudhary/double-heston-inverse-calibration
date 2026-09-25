import numpy as np
from scripts.mentor_dh_pinn.prepare_regular_pinn_surfaces import preconditioner
from scripts.mentor_dh_pinn.diagnose_regular_pinn_validation import exact_iv_jacobian
from src.mentor_dh_pinn.regular_pinn_data import teacher_labels


def test_tolerance_coordinate_inverse_and_sensitivity_target():
    unit=np.array([.4,.6,.5,.45,.65]);x=np.tile(-np.log(np.linspace(.8,1.2,21)),6)
    tau=np.repeat(np.array([30,60,90,180,365,730])/365,21)
    q=np.column_stack([x,np.log(tau),np.broadcast_to(unit,(126,5))])
    iv,j=exact_iv_jacobian(q,1,128)
    b,s,rank,a=preconditioner(j,unit,1,absolute_floor=1e-12)
    assert rank==5
    np.testing.assert_allclose(b@a,np.eye(5),atol=1e-9)
    h=1e-5;plus=q.copy();minus=q.copy();plus[:,4]+=h;minus[:,4]-=h
    ip=np.sqrt(teacher_labels(plus,1,gradients=False)['w']/tau)
    im=np.sqrt(teacher_labels(minus,1,gradients=False)['w']/tau)
    np.testing.assert_allclose((ip-im)/(2*h),j[:,2],atol=1e-7,rtol=1e-4)
    dropped,_,rank,_=preconditioner(j,unit,1,absolute_floor=s[0]*2)
    assert rank==0 and np.count_nonzero(dropped)==0
