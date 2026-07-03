<template>
  <div class="app" :class="{ 'app--layout': !noLayout }">
    <AppSidebar v-if="!noLayout" />
    <main class="app-main">
      <router-view v-slot="{ Component }">
        <template v-if="$route.meta.keepAlive">
          <keep-alive>
            <component :is="Component" />
          </keep-alive>
        </template>
        <template v-else>
          <component :is="Component" />
        </template>
      </router-view>
    </main>
  </div>
</template>

<script setup>
import { computed } from 'vue'
import { useRoute } from 'vue-router'
import AppSidebar from './components/AppSidebar.vue'

const route = useRoute()
const noLayout = computed(() => route.meta.noLayout)
</script>

<style>
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

html, body {
  font-size: 16px;
  height: 100%;
  width: 100%;
  overflow: hidden;
}

.app {
  height: 100vh;
  width: 100vw;
}

.app--layout {
  display: flex;
}

.app-main {
  flex: 1;
  height: 100vh;
  overflow-y: auto;
  background: var(--color-bg);
}
</style>
