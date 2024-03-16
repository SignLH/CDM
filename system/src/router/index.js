import Vue from 'vue'
import VueRouter from 'vue-router'
import {buildConfig} from '@/settings'
import Layout from '@/layout'
import {SceneNameAssemble} from '@/utils/services/overlay/overlay.model'

Vue.use(VueRouter)

const routes = [
    {
        path: '/',
        component: Layout,
        redirect: '/home',
        children: [{
            path: '/home',
            component: () => import('@/views/home/index'),
            name: 'home',
            meta: {
                pageNav: true,
                needLogin: true,
                name: "大屏首页",
                viewName: SceneNameAssemble.default
            }
        },
        ]
    },
]


const router = new VueRouter({
    mode: 'hash',
    base: buildConfig.base,
    routes
})

export default router
