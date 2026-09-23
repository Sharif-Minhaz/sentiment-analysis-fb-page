<script setup>
defineProps({ model: { type: Object, default: () => ({}) } })
</script>

<template>
  <section v-if="model.top_features" class="card p-4">
    <h2 class="text-sm font-medium mb-1">What the model learned</h2>
    <p class="text-xs mb-3" :style="{ color: 'var(--muted)' }">
      Logistic regression assigns one weight per feature, so you can read the model
      directly - these are the strongest weights in each direction.
    </p>

    <div class="grid md:grid-cols-2 gap-4 mb-4">
      <div v-for="side in [
        { key: 'positive', label: 'Pushes positive', color: 'var(--positive)' },
        { key: 'negative', label: 'Pushes negative', color: 'var(--negative)' },
      ]" :key="side.key">
        <p class="text-xs mb-2 flex items-center gap-2">
          <span class="w-2.5 h-2.5 rounded-sm" :style="{ background: side.color }" />
          <span :style="{ color: 'var(--ink-2)' }">{{ side.label }}</span>
        </p>
        <div class="flex flex-wrap gap-1.5">
          <span
            v-for="feature in model.top_features[side.key]"
            :key="feature"
            class="text-xs px-2 py-1 rounded border"
            :style="{ borderColor: 'var(--hairline)' }"
          >{{ feature }}</span>
        </div>
      </div>
    </div>

    <dl
      class="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm pt-3 border-t"
      :style="{ borderColor: 'var(--hairline)' }"
    >
      <div>
        <dt class="text-xs" :style="{ color: 'var(--muted)' }">Config</dt>
        <dd>{{ model.config_name }}</dd>
      </div>
      <div>
        <dt class="text-xs" :style="{ color: 'var(--muted)' }">Trained on</dt>
        <dd class="num">{{ model.rows_used?.toLocaleString() }} rows</dd>
      </div>
      <div>
        <dt class="text-xs" :style="{ color: 'var(--muted)' }">Test accuracy</dt>
        <dd class="num">{{ (model.test_metrics?.accuracy * 100).toFixed(2) }}%</dd>
      </div>
      <div>
        <dt class="text-xs" :style="{ color: 'var(--muted)' }">Neutral band</dt>
        <dd class="num">{{ model.neutral_band?.join(' – ') }}</dd>
      </div>
    </dl>
  </section>
</template>
