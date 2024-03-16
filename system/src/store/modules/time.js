import {SceneNameAssemble} from "@/utils/services/overlay/overlay.model";
import {setCameraStatus} from "@/utils/services/camera";
import {mapService} from "@/utils/app-base";
import {updateEntityByGroup} from "@/utils/services/overlay/entity";

const state = {
  currentTime: new Date(),
  intervalHandler: null,
  //当前场景
  currentScene: SceneNameAssemble.morning
}

const mutations = {
  Set_Current_Time: (state, time) => {
    state.currentTime = time
  },
  Set_Current_Scene: (state, v) => {
    state.currentScene = v
  },
}

const actions = {
  /**
   * 开始监控时间
   * @param state
   * @param commit
   * @param dispatch
   * @param timeout
   */
  startTimeInterval({state, commit, dispatch}, timeout = 1000) {
    if (state.intervalHandler != null) {
      return
    }
    console.log('开始实时设置时间')
    state.intervalHandler = setInterval(() => {
      commit('Set_Current_Time', new Date())
    }, timeout)
  },
  /**
   * 结束监控时间
   * @param state
   * @param commit
   */
  clearTimeInterval({state, commit}) {
    if (state.intervalHandler == null) {
      return
    }
    clearInterval(state.intervalHandler)
    state.intervalHandler = null
  },
}

export default {
  namespaced: true,
  state,
  mutations,
  actions
}
