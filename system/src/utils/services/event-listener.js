import { map, filter } from 'rxjs';
import LOG from "@/utils/self-log";

let subscription=null

// 注册全局鼠标事件
export let clickEventListener = []

// 注册全局滚轮事件
export let wheelEventListener = []

export function initEventListener(client) {
  client.addEventListener('click', (event) => {
    // LOG.info(['event',event,'clickEventListener',clickEventListener)
    // const data = event.detail //data.entityId; 通过id判断哪个点击
    // LOG.info(['clickEvent',data])
    clickEventListener.forEach(funC => funC(event))
  });


  client.addEventListener('wheel', (event) => {
    LOG.info(['wheelEvent',event])
    wheelEventListener.forEach(funC => funC(event))
  });


// 从消息总线中过滤出事件总线，建议仅生成全局唯一的事件总线
//   const eventBus = client.adapter.messageBus.pipe(
//     map((msg) => msg.payload.params ),
//     filter((params) => params?.event),
//   );

// --------------------------------------------------------
// 侦听方式一：直接订阅全局事件总线
// 该方式建议仅全局唯一订阅，所有的事件类型都会被统一函数侦听，在函数中再进行逻辑区分
//   subscription = eventBus.subscribe((params) => {
//     // 解析 params 参数，根据需要侦听的事件类型进行过滤，params 参数详见每个事件类型的说明
//     // params 的参数格式如下：
//     // {
//     //     event: "XXXXX",  // 必有，触发事件字符串，主要通过该参数筛选过滤
//     //     type: "XXXXXX"   // 必有，事件类型字符串，可结合该参数筛选过滤
//     //     target: "XXXX",  // 必有，目标标识，可结合该参数筛选过滤，如"global"可筛选全局事件
//     //     detail: {}       // 根据事件类型可能有，主要的返回信息，解析后用于后续业务处理
//     // }
//     LOG.info(['eventBus',params=)
//     const type=params.type
//     switch (params.event) {
//       case "MouseEvent":
//         if (type==='click') {
//           clickEventListener.forEach(funC => funC(params))
//         }else if (type==='wheel') {
//           wheelEventListener.forEach(funC => funC(params))
//         }
//         break
//     }
//   })
}


// 当业务逻辑处理完成时，可及时取消事件订阅
export function clearSubscription(){
  subscription&&subscription.unsubscribe()
}
