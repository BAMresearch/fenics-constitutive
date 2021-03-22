Thoughts/design
===============

Performance
-----------

Unless proven otherwise, the C++ code is not expected to be the bottleneck of 
your simulation. This is especially true for anything involving a "solve".

Reuseability
------------

Try to break down the building blocks of your material law to be reusable by 
others. If you are able to decompose e.g. a norm

.. math::
   \|y(\boldsymbol x)\| = A \  \mathrm{tr}(\boldsymbol x) + \ldots \\
   \frac{\partial \|y\|}{\partial \boldsymbol x} = \ldots

into a function of invariants :math:`I1, J2, \ldots`, please do so! This makes
it reuseable independent of the specific notation (Mandel, Voigt, Nye, Tensor)
or dimension of :math:`\boldsymbol x`, as long as it defines those invariants.

Then, derivatives easily chain to

.. math::
   \frac{\partial \|y\|}{\partial \boldsymbol x} = \frac{\partial \|y(I1(\boldsymbol x), J2(\boldsymbol x))\|}{I1} \frac{\partial I1}{\partial \boldsymbol x} + \ldots

Same is true for the invariants itself. They may only be implemented for a 3D 
tensor version of :math:`\boldsymbol x` and transformations from, e.g.  2D 
plane strain to this notation is provided.

Question
^^^^^^^^

Instead of 

.. code-block:: python

   eeq, deeq = some_norm_2d(e2d)

you'd do

.. code-block:: python

   e3d = T_to_3d @ e2d
   eeq = some_norm_2d(e3d.I1, e3d.J2)
   deeq_3d = some_norm_2d.dI1(e3d.I1, e3d.J2) * e3d.dI1 \ 
           + some_norm_2d.dJ2(e3d.I1, e3d.J2) * e3d.dJ2
   deeq = T_to_3d.T @ deeq_3d


Unless that is somehow automated... is that worth it? Maybe the latter should
only be used for testing.

Formats
-------

FEniCS
^^^^^^

.. code-block:: python

   QHandler?
   Connects c++ object with q object

