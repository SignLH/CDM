import LOG from "@/utils/self-log";


// key function
const TimingTask = new Map()
let TimingTaskInterval = null

/**
 * 定时任务
 * @param key
 * @param func 函数不可使用过于复杂任务 为空 删除key
 */
export function setTimingTask(key , func) {
  if (key&&func) {
    TimingTask.set(key, func)
  }else if (key){
    TimingTask.delete(key)
  }
}

//初始化定时任务
export function InitTimingTask(speed = 90) {
  if (TimingTaskInterval) {
    return
  }
  TimingTaskInterval = setInterval(() => {
    //执行注册方法
    TimingTask.forEach((v, key) => {
      try {
        v()
      } catch (e) {
        LOG.warn([e, '执行定义任务失败', key])
      }
    })
  }, speed)
}

/**
 * 获取随机唯一字符串 len 个随机字符串 + timestamp
 * @return {string}
 */
export function getRandomKey(len=5) {
  let $chars = 'ABCDEFGHJKMNOPQRSTUVWXYZabcdefghijkmnoprstuvwxyz123456789';
  let maxPos = $chars.length;
  let pwd = '';
  for (let i = 0; i < len; i++) {
    pwd += $chars.charAt(Math.floor(Math.random() * maxPos));
  }
  pwd+=new Date().getTime()
  return pwd;
}
