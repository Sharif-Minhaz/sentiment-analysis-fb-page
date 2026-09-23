<script setup>
import { percentOf } from '../format'

defineProps({ posts: { type: Array, default: () => [] } })
</script>

<template>
  <section v-if="posts.length" class="card p-4">
    <h2 class="text-sm font-medium mb-3">By post</h2>

    <div class="overflow-x-auto">
      <table class="w-full text-sm">
        <thead>
          <tr class="text-left text-xs uppercase tracking-wide" :style="{ color: 'var(--muted)' }">
            <th class="pb-2 font-medium">Post</th>
            <th class="pb-2 font-medium text-right">Total</th>
            <th class="pb-2 font-medium text-right">Pos</th>
            <th class="pb-2 font-medium text-right">Neu</th>
            <th class="pb-2 font-medium text-right">Neg</th>
            <th class="pb-2 pl-3 font-medium w-32">Mix</th>
          </tr>
        </thead>
        <tbody class="divide-y divide-[color:var(--hairline)]">
          <tr v-for="post in posts" :key="post.id">
            <td class="py-2 pr-3 max-w-xs truncate">{{ post.message || post.id }}</td>
            <td class="py-2 text-right num">{{ post.total }}</td>
            <td class="py-2 text-right num">{{ post.positive }}</td>
            <td class="py-2 text-right num">{{ post.neutral }}</td>
            <td class="py-2 text-right num">{{ post.negative }}</td>
            <td class="py-2 pl-3">
              <!-- 2px surface gaps between segments, matching the chart marks -->
              <div class="flex h-2 rounded-full overflow-hidden" :style="{ background: 'var(--grid)' }">
                <div :style="{ width: percentOf(post.negative, post.total), background: 'var(--negative)', borderRight: '2px solid var(--surface)' }" />
                <div :style="{ width: percentOf(post.neutral, post.total), background: 'var(--neutral)', borderRight: '2px solid var(--surface)' }" />
                <div :style="{ width: percentOf(post.positive, post.total), background: 'var(--positive)' }" />
              </div>
            </td>
          </tr>
        </tbody>
      </table>
    </div>
  </section>
</template>
