<template>
  <aside class="app-sidebar">
    <div class="sidebar-top">
      <div class="sidebar-brand">
        <div class="brand-icon">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
            <path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/>
            <path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/>
          </svg>
        </div>
        <span class="brand-name">AI Second Brain</span>
      </div>

      <!-- 新建会话按钮 -->
      <button class="btn-new-chat" @click="startNewChat">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <line x1="12" y1="5" x2="12" y2="19"/>
          <line x1="5" y1="12" x2="19" y2="12"/>
        </svg>
        <span>新建会话</span>
      </button>

      <!-- 导航区 -->
      <nav class="sidebar-nav">
        <router-link
          v-for="item in navItems"
          :key="item.path"
          :to="item.path"
          class="nav-item"
          :class="{ active: isActive(item.path) }"
        >
          <span class="nav-icon" v-html="item.icon"></span>
          <span class="nav-label">{{ item.label }}</span>
        </router-link>
      </nav>

      <!-- 会话历史 -->
      <div class="session-section">
        <div class="session-section-title">会话历史</div>
        <div class="session-list" v-if="sessionStore.sessions.length > 0">
          <div
            v-for="session in sessionStore.sessions"
            :key="session.session_id"
            class="session-item"
            :class="{ active: isSessionActive(session) }"
            @click="selectSession(session)"
          >
            <svg class="session-icon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
              <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
            </svg>
            <span class="session-title">{{ session.title || '新会话' }}</span>
            <button class="session-delete" @click.stop="deleteSession(session)" title="删除会话">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
                <polyline points="3 6 5 6 21 6"/>
                <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>
              </svg>
            </button>
          </div>
        </div>
        <div v-else class="session-empty">
          <span>暂无历史会话</span>
        </div>
      </div>
    </div>

    <div class="sidebar-footer">
      <div class="user-block" @click="goToProfile">
        <div class="user-avatar">
          <span v-if="!userStore.userInfo?.avatar">{{ avatarLetter }}</span>
          <img v-else :src="userStore.userInfo.avatar" alt="avatar" />
        </div>
        <div class="user-info">
          <div class="user-name">{{ displayName }}</div>
          <div class="user-bio">{{ displayBio }}</div>
        </div>
      </div>
      <div class="settings-wrapper">
        <button class="settings-btn" @click.stop="togglePopover">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
            <circle cx="12" cy="12" r="1"/><circle cx="12" cy="5" r="1"/><circle cx="12" cy="19" r="1"/>
          </svg>
        </button>
        <SettingsPopover v-if="popoverVisible" @close="popoverVisible = false" />
      </div>
    </div>
  </aside>
</template>

<script setup>
import { ref, computed, onMounted, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useUserStore } from '../store/user'
import { useSessionStore } from '../store/session'
import SettingsPopover from './SettingsPopover.vue'
import { showToast } from 'vant'

const route = useRoute()
const router = useRouter()
const userStore = useUserStore()
const sessionStore = useSessionStore()
const popoverVisible = ref(false)

const navItems = [
  {
    path: '/notes',
    label: '笔记',
    icon: '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/><polyline points="10 9 9 9 8 9"/></svg>'
  },
  {
    path: '/review',
    label: '回顾',
    icon: '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/><path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/></svg>'
  },
  {
    path: '/knowledge',
    label: '知识库',
    icon: '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/><line x1="12" y1="6" x2="12" y2="12"/><line x1="9" y1="9" x2="15" y2="9"/></svg>'
  }
]

function isActive(path) {
  if (path === '/notes') return route.path.startsWith('/notes')
  if (path === '/review') return route.path.startsWith('/review')
  if (path === '/knowledge') return route.path.startsWith('/knowledge')
  return false
}

function isSessionActive(session) {
  return route.path.startsWith('/chat') && sessionStore.currentSession?.session_id === session.session_id
}

function startNewChat() {
  sessionStore.currentSession = null
  router.push('/chat')
}

function selectSession(session) {
  sessionStore.setCurrentSession(session)
  router.push(`/chat/${session.session_id}`)
}

async function deleteSession(session) {
  const result = await sessionStore.deleteSession(session.session_id)
  if (result.success) {
    showToast('会话已删除')
    // 如果删除的是当前会话，回到聊天首页
    if (sessionStore.currentSession?.session_id === session.session_id) {
      sessionStore.currentSession = null
      if (route.path.startsWith('/chat')) {
        router.push('/chat')
      }
    }
  } else {
    showToast(result.message || '删除失败')
  }
}

async function loadSessions() {
  if (!userStore.getLoginStatus) return
  if (!userStore.userInfo) {
    await userStore.getUserInfoDetail()
  }
  if (userStore.userInfo) {
    const userId = userStore.userInfo.uuid || userStore.userInfo.id || userStore.userInfo.user_id
    if (userId) {
      await sessionStore.getUserSessions(userId)
    }
  }
}

onMounted(() => {
  loadSessions()
})

// 路由切换到 /chat/:sessionId 时刷新会话列表（新会话创建后自动追加到侧边栏）
watch(() => route.params.sessionId, (newId, oldId) => {
  if (newId && newId !== oldId) {
    loadSessions()
  }
})

const avatarLetter = computed(() => {
  const name = userStore.userInfo?.username
  return name ? name.charAt(0).toUpperCase() : '?'
})

const displayName = computed(() => {
  return userStore.userInfo?.username || '未登录'
})

const displayBio = computed(() => {
  return userStore.userInfo?.bio || userStore.userBio || '点击设置个人简介'
})

function goToProfile() {
  if (userStore.isLogin) {
    router.push('/profile')
  } else {
    router.push('/login')
  }
}

function togglePopover() {
  popoverVisible.value = !popoverVisible.value
}
</script>

<style scoped>
.app-sidebar {
  width: 260px;
  height: 100vh;
  background-color: var(--color-sidebar);
  display: flex;
  flex-direction: column;
  flex-shrink: 0;
  border-right: 1px solid var(--color-border-light);
  position: relative;
  z-index: 10;
}

.sidebar-top {
  flex: 1;
  display: flex;
  flex-direction: column;
  padding: 20px 12px 12px;
  overflow: hidden;
}

.sidebar-brand {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 4px 12px 16px;
}

.brand-icon {
  width: 32px;
  height: 32px;
  border-radius: 8px;
  background: var(--color-primary);
  display: flex;
  align-items: center;
  justify-content: center;
  color: #fff;
}

.brand-name {
  font-family: var(--font-heading);
  font-size: 15px;
  font-weight: 600;
  color: var(--color-text);
}

/* ---- 新建会话按钮 ---- */
.btn-new-chat {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  width: 100%;
  padding: 12px 16px;
  margin-bottom: 8px;
  border-radius: 999px;
  border: none;
  background: var(--color-primary);
  color: #fff;
  font-size: 14px;
  font-weight: 500;
  cursor: pointer;
  transition: opacity 0.2s, transform 0.1s;
}

.btn-new-chat:hover {
  opacity: 0.88;
}

.btn-new-chat:active {
  transform: scale(0.98);
}

/* ---- 导航 ---- */
.sidebar-nav {
  display: flex;
  flex-direction: column;
  gap: 2px;
  margin-bottom: 16px;
}

.nav-item {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 11px 14px;
  border-radius: 8px;
  color: var(--color-text-light);
  text-decoration: none;
  font-size: 14px;
  transition: background 0.15s, color 0.15s;
  position: relative;
}

.nav-item:hover {
  background: var(--color-surface);
  color: var(--color-text);
}

.nav-item.active {
  background: var(--color-surface);
  color: var(--color-primary);
  font-weight: 500;
}

.nav-item.active::before {
  content: '';
  position: absolute;
  left: 0;
  top: 8px;
  bottom: 8px;
  width: 3px;
  border-radius: 0 3px 3px 0;
  background: var(--color-primary);
}

.nav-icon {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 20px;
  height: 20px;
}

.nav-label {
  white-space: nowrap;
}

/* ---- 会话历史 ---- */
.session-section {
  flex: 1;
  display: flex;
  flex-direction: column;
  min-height: 0;
}

.session-section-title {
  font-size: 12px;
  font-weight: 500;
  color: var(--color-text-lighter);
  padding: 8px 12px 6px;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.session-list {
  flex: 1;
  overflow-y: auto;
  margin: 0 -4px;
}

.session-item {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 9px 12px;
  margin: 1px 4px;
  border-radius: 8px;
  cursor: pointer;
  transition: background 0.15s;
  position: relative;
}

.session-item:hover {
  background: var(--color-surface);
}

.session-item.active {
  background: var(--color-surface);
}

.session-item.active .session-title {
  color: var(--color-primary);
  font-weight: 500;
}

.session-icon {
  flex-shrink: 0;
  color: var(--color-text-lighter);
}

.session-item.active .session-icon {
  color: var(--color-primary);
}

.session-title {
  flex: 1;
  font-size: 13px;
  color: var(--color-text-light);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  min-width: 0;
}

.session-delete {
  flex-shrink: 0;
  display: none;
  align-items: center;
  justify-content: center;
  width: 24px;
  height: 24px;
  border-radius: 4px;
  border: none;
  background: transparent;
  color: var(--color-text-lighter);
  cursor: pointer;
  transition: color 0.15s, background 0.15s;
}

.session-item:hover .session-delete {
  display: flex;
}

.session-delete:hover {
  color: #e05544;
  background: rgba(224, 85, 68, 0.1);
}

.session-empty {
  padding: 16px 12px;
  text-align: center;
  font-size: 12px;
  color: var(--color-text-lightest);
}

/* ---- 底部用户区 ---- */
.sidebar-footer {
  padding: 12px;
  border-top: 1px solid var(--color-border-light);
  display: flex;
  align-items: center;
  gap: 8px;
}

.user-block {
  flex: 1;
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 8px;
  border-radius: 8px;
  cursor: pointer;
  transition: background 0.15s;
  min-width: 0;
}

.user-block:hover {
  background: var(--color-surface);
}

.user-avatar {
  width: 34px;
  height: 34px;
  border-radius: 50%;
  background: var(--color-surface);
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
  font-size: 14px;
  font-weight: 600;
  color: var(--color-primary);
  overflow: hidden;
}

.user-avatar img {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.user-info {
  flex: 1;
  min-width: 0;
}

.user-name {
  font-size: 13px;
  font-weight: 500;
  color: var(--color-text);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.user-bio {
  font-size: 11px;
  color: var(--color-text-lighter);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  margin-top: 1px;
}

.settings-wrapper {
  position: relative;
}

.settings-btn {
  width: 32px;
  height: 32px;
  border-radius: 6px;
  border: none;
  background: transparent;
  color: var(--color-text-lighter);
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: background 0.15s, color 0.15s;
}

.settings-btn:hover {
  background: var(--color-surface);
  color: var(--color-text);
}
</style>
