#!/usr/bin/env bash
set -euo pipefail

# Parámetros de grid
epochs=100

export LC_NUMERIC=C
alphas=(0.9)
betas=(0.8)


max_runs=10

counter=0

CFG_DIR="config"
CFG_BASE="${CFG_DIR}/config.yaml"
TMP_CFG="${CFG_DIR}/tmp_config.yaml"

for alpha in "${alphas[@]}"; do
  for beta in "${betas[@]}"; do
    if (( counter >= max_runs )); then
      break 2
    fi

    echo
    echo "🔧 Entrenando: alpha=${alpha}, beta=${beta}, epochs=${epochs}"

    out="runs/alpha_${alpha}_beta_${beta}_epochs_${epochs}"
    mkdir -p "${out}/"{models,logs,plots,checkpoints}

    # 1) Copiar y parchear config
    cp "${CFG_BASE}" "${TMP_CFG}"
    sed -i "s/^[[:space:]]*alpha:.*/  alpha: ${alpha}/"          "${TMP_CFG}"
    sed -i "s/^[[:space:]]*beta:.*/  beta: ${beta}/"            "${TMP_CFG}"
    sed -i "s/^[[:space:]]*epochs:.*/  epochs: ${epochs}/"      "${TMP_CFG}"
    sed -i "s|^[[:space:]]*checkpoint_dir:.*|  checkpoint_dir: ${out}/checkpoints|" "${TMP_CFG}"
    sed -i "s|^[[:space:]]*output_file:.*|  output_file: ${out}/logs/performance_log.csv|" "${TMP_CFG}"

    export ALPHA="${alpha}"
    export BETA="${beta}"

    # 2) Ejecutar entrenamiento usando el tmp_config
    CONFIG_PATH="${TMP_CFG}" python -m scripts.main 2>&1 | tee "${out}/logs/full_output.log"

    # 3) Extraer ruta del checkpoint final
    best=$(grep -F "🔖 Modelo final guardado en" -m1 "${out}/logs/full_output.log" | awk '{print $NF}')
    if [[ ! -f "${best}" ]]; then
      echo "❌ No encontré el modelo final en stdout."
      exit 1
    fi

    mv "${best}" "${out}/models/"

    # 4) Mover métricas y curvas si existen
    mv logs/training_metrics.csv "${out}/logs/"   || true
    mv plots/training_curves.png   "${out}/plots/" || true

    echo "✅ Resultados en ${out}"
    ((counter++))
  done
done

rm -f "${TMP_CFG}"
echo
echo "🏁 Entrenamientos completados: ${counter} runs."
