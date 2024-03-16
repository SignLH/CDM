import {clearSubscription, initEventListener} from "@/utils/services/event-listener";
import {deleteEntityByGroup} from "@/utils/services/overlay/entity";
import LOG from "@/utils/self-log";

export async function clientInit(client) {
  //删除所有覆盖物
  await deleteEntityByGroup(client,'/MEntity')
  //重启分辨率
  await client.setResolution({width: 1920, height: 1080})

  LOG.info(['删除所有覆盖物，重启分辨率{width: 1920, height: 1080} 完成！'])
  //--https://diva.sheencity.cn/docs/index.html#/sdk/global
  initEventListener(client)

}

export async function clientClear(client) {
  clearSubscription()
}
