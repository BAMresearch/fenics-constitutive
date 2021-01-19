# Basic IP based constitutive laws

* They do not really share a common interface. 
* Material parameters are passed to constructor.
* Nice for testing.


### LinearElastic/NonlinearElastic
* history variables:
    * None
* methods:
    * stress, dstress = eval(strain)

### LocalDamage
* history variables:
    * scalar kappa
* methods:
    * stress, dstress, kappa_new = stress(strain, kappa)
* specials:
    * kappa, dkappa = evolution_kappa(strain, kappa)
    * omega, domega = damage_law(kappa)

### Plasticity
* history variables:
    * plastic strain tensor
    * scalar plastic multiplyier lambda
* methods:
    * stress, dstress, eps_pl_new, lambda_new = evaluate(strain, eps_pl, lambda)
* specials:
    * yield function and hardening rule should be replacable


# Structural constitutive laws

* Share a common mechanics (stress-dstress) interface.
* Provide and handle the returned history data to IP laws
* Loop over IPs


### StructureConstitutiveLaw
* history variables:
    * all
* methods:
    * void eval(strain, ip_id)
    * void eval(strain, other_dof, ip_id)

