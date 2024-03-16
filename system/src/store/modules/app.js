import {
    getSceneInfo
} from '@/api/login'
const state = {
    //normal 1920; big: 3000+
    size: localStorage.getItem('PageSize') || 'normal',

    //当前页面name
    pageName: '',

    //左右界面是否打开
    leftOpen: true,
    rightOpen: true,

    //大屏数据id集合
    screenIds: [],

    //页面内存储的重点监控列表
    pageMonitorList: []
}

function setAnimation(elementData, tranSize = 520) {
    if (elementData == null) {
        // console.log('setAnimation elementData is null')
        return
    }
    const styleStr = `transition: all .5s linear;transform: translateX(${tranSize}px);`
    elementData.setAttribute('style', styleStr)
}

const mutations = {

    SET_SIZE: (state, size) => {
        state.size = size
        localStorage.getItem('PageSize', size)
    },
    SET_PAGE_NAME: (state, pageName) => {
        state.pageName = pageName
    },

    SET_LEFT_OPEN: (state, leftOpen) => {
        state.leftOpen = leftOpen
        //左窗口
        const elementsByClassName = document.getElementsByClassName('page-left-view')
        for (let elementsByClassNameElement of elementsByClassName) {
            if (leftOpen) {
                elementsByClassNameElement.classList.remove('animation-left-out')
                elementsByClassNameElement.classList.add('animation-left-in')
            } else {
                elementsByClassNameElement.classList.remove('animation-left-in')
                elementsByClassNameElement.classList.add('animation-left-out')
            }
        }
        //frame-view
        const frameLeft = document.getElementById('frame-left')
        setAnimation(frameLeft, leftOpen ? 520 : 0)
        setAnimation(document.getElementById('direction-nav'), leftOpen ? 520 : 0)
        setAnimation(document.getElementById('area-nav'), leftOpen ? 520 : 0)
    },
    SET_RIGHT_OPEN: (state, rightOpen) => {
        state.rightOpen = rightOpen
        const elementsByClassName = document.getElementsByClassName('page-right-view')
        for (let elementsByClassNameElement of elementsByClassName) {
            if (rightOpen) {
                elementsByClassNameElement.classList.remove('animation-right-out')
                elementsByClassNameElement.classList.add('animation-right-in')
            } else {
                elementsByClassNameElement.classList.remove('animation-right-in')
                elementsByClassNameElement.classList.add('animation-right-out')
            }
        }
        //frame-view
        const frameRight = document.getElementById('frame-right')
        setAnimation(frameRight, rightOpen ? -520 : 0)
        setAnimation(document.getElementById('map-nav'), rightOpen ? -520 : 0)
        setAnimation(document.getElementById('map-nav2'), rightOpen ? -520 : 0)
    },
    SET_SCREEN_IDS: (state, ids) => {
        state.screenIds = ids
    },
    SET_PAGE_MONITOR_LIST: (state, list) => {
        state.pageMonitorList = list
    },

}

const actions = {

    setSize({
        commit
    }, size) {
        commit('SET_SIZE', size)
    },
    setPageName({
        commit
    }, name) {
        commit('SET_PAGE_NAME', name)
    },
    setOpenStatus({
        commit
    }, obj) {
        return new Promise((resolve, reject) => {
            if (obj.key == 'ALL') {
                commit('SET_LEFT_OPEN', obj.open)
                commit('SET_RIGHT_OPEN', obj.open)
                setTimeout(() => {
                    resolve()
                }, 300)
            } else if (obj.key == 'LEFT') {
                commit('SET_LEFT_OPEN', obj.open)
                resolve()
            } else if (obj.key == 'RIGHT') {
                commit('SET_RIGHT_OPEN', obj.open)
                resolve()
            }
        })
    },
    getScreenIds({
        commit
    }) {
        return new Promise((resolve, reject) => {
            getSceneInfo().then(res => {
                let result = res.result || []
                commit('SET_SCREEN_IDS', result)
                resolve(result)
            })
        })
    },

    setPageMonitorList({
        commit
    }, list) {
        commit('SET_PAGE_MONITOR_LIST', list)
    },
}

export default {
    namespaced: true,
    state,
    mutations,
    actions
}
