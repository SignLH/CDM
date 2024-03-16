import {getImageBaseUrl} from "@/utils/app-base";

//获取真实图片数据
export function getTrueImageList(listData=[]){
  let res=[]
  listData.forEach(e=>{
    if (e){
      res.push(getImageBaseUrl()+e)
    }
  })

  return res;
}
