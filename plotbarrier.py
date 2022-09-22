import numpy as np
import matplotlib.pyplot as plt

Init = np.load('Init.npy')
Init_next = np.load('Init_next.npy')

Unsafe = np.load('Unsafe.npy')
Unsafe_next = np.load('Unsafe_next.npy')

Lie = np.load('Lie.npy')
Lie_next = np.load('Lie_next.npy')

Init_final = np.hstack((Init, Init_next[:50]))
Unsafe_final = np.hstack((Unsafe, Unsafe_next[:50]))
Lie_final = np.hstack((Lie, Lie_next[:50]))

plt.plot(list(range(len(Lie_final))),Lie_final, label='Lie')
plt.plot(list(range(len(Unsafe_final))), Unsafe_final, label='$\min_{x_{u} \in X_u} B(x_{u})$')
plt.plot(list(range(len(Init_final))), Init_final, label='$\max_{x_{0} \in X_0} B(x_{0})$')
plt.legend()
plt.savefig('rocket_barrier_loss.pdf', bbox_inches='tight')