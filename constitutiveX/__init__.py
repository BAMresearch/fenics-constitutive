
from pathlib import Path
import git

from constitutiveX.cpp import (jaumann_rotate_3d,jaumann_rotate_fast_3d, mandel_to_tensor_3d, 
                                tensor_to_mandel_3d, Constraint, Q, LinearElastic3D, JH23D, JH2Nonlocal3D, JH2Parameters, HypoElastic3D, apply_b_bar_3d)
from constitutiveX.explicit_dynamics import CDMPlaneStrainX, ImplicitNonlocalVariable
#import constitutiveX.cpp as cpp

def git_version():
    git_path = Path(__file__).parents[1] / ".git"
    repo = git.Repo(git_path)
    hash = repo.head.commit.hexsha
    if repo.is_dirty():
        hash = "dirty+"+hash
    return hash

__githash__ = git_version()
