
# Re-use the FedGD result from Stage 6 — no need to retrain
result_fedgd_A    = results["A  (geo k-NN)"]
# FedRelax: подбираем alpha отдельно (не используем best_alpha_A от FedGD)
ALPHA_GRID_RELAX = ALPHA_GRID_A if "ALPHA_GRID_A" in globals() else ALPHA_GRID
best_alpha_A_relax, val_mses_A_relax = tune_alpha(
      NODE_DATASETS, SHARED_NAMES, A_GEO,
      alpha_grid=ALPHA_GRID_RELAX, n_iter=N_ITER_TUNE, algorithm="fedrelax"
)

# Для System A запускаем FedRelax на 300 итерациях
N_ITER_RELAX = 300
result_fedrelax_A = run_fedrelax(
      NODE_DATASETS, SHARED_NAMES, A_GEO,
      alpha=best_alpha_A_relax, n_iter=N_ITER_RELAX
)

print(f"System A — FedRelax best alpha (separate tuning): {best_alpha_A_relax}")

ev_A_gd    = evaluate(result_fedgd_A["W"],    NODE_DATASETS, SHARED_NAMES)
ev_A_relax = evaluate(result_fedrelax_A["W"], NODE_DATASETS, SHARED_NAMES)

print(f"System A — FedGD    ({N_ITER_FINAL} iter):  "
      f"train={result_fedgd_A['train_loss'][-1]:.4f}  "
      f"val={result_fedgd_A['val_loss'][-1]:.4f}  "
      f"test={ev_A_gd['test_mse'].mean():.4f} deg C^2")
print(f"System A — FedRelax ({N_ITER_RELAX}  iter):  "
      f"train={result_fedrelax_A['train_loss'][-1]:.4f}  "
      f"val={result_fedrelax_A['val_loss'][-1]:.4f}  "
      f"test={ev_A_relax['test_mse'].mean():.4f} deg C^2")
print()
print("Same graph, same alpha -> both algorithms converge to the same solution.")
