import {formatDate} from "@/utils/index";
import {AppBaseConfig} from "@/utils/app-base";

//${data.startTime} 00:00:00  {data.endTime} 23:59:59  返回时间
export function getStartEndTime(timeList, startTimeProp = 'startTime', endTimeProp = 'endTime') {
  if (timeList == null || timeList.length === 0) {
    return {}
  }
  return {
    [startTimeProp]: `${timeList[0]} 00:00:00`,
    [endTimeProp]: `${timeList[1]} 23:59:59`,
  }
}

/**
 * JavaScript 中的 Date 对象和一些方法来获取当前月份的第一天和最后一天的时间戳
 * @param formatStr
 * @returns {Date[]} or {String[]}
 */
export function getCurrentMonthFirstLastDay(formatStr) {
  const now = new Date();
  let firstDayOfMonth = new Date(now.getFullYear(), now.getMonth(), 1);
  let lastDayOfMonth = new Date(now.getFullYear(), now.getMonth() + 1, 0);
  if (formatStr) {
    firstDayOfMonth = formatDate(firstDayOfMonth, formatStr)
    lastDayOfMonth = formatDate(lastDayOfMonth, formatStr)
  }

  return [firstDayOfMonth, lastDayOfMonth]
}


// ------------------------------------------------------color
const ColorList = [
  '#9E87FF', '#73DDFF', '#fe9a8b', '#F56948', '#9E87FF', '#006400',
  '#7FFF00', '#FF1493', '#C71585', '#191970', '#FF00FF', '#90EE90',
  '#B0C4DE', '#00FF7F', '#008080', '#BDB76B', '#00FFFF', '#6B8E23',

  // '#FFB6C1', '#FFC0CB', '#DC143C', '#FFF0F5', '#DB7093', '#FF69B4',
  // '#DA70D6', '#D8BFD8', '#DDA0DD', '#EE82EE',
  // '#FF00FF', '#8B008B', '#800080', '#BA55D3', '#9400D3',
  // '#9932CC', '#4B0082', '#8A2BE2', '#9370DB', '#7B68EE', '#6A5ACD',
  // '#483D8B', '#E6E6FA', '#F8F8FF', '#0000FF', '#0000CD',
  // '#00008B', '#000080', '#4169E1', '#6495ED',  '#778899',
  // '#708090', '#1E90FF', '#F0F8FF', '#4682B4', '#87CEFA', '#87CEEB',
  // '#00BFFF', '#ADD8E6', '#B0E0E6', '#5F9EA0', '#F0FFFF', '#E1FFFF',
  // '#AFEEEE', '#00FFFF', '#00FFFF', '#00CED1', '#2F4F4F', '#008B8B',
  //  '#48D1CC', '#20B2AA', '#40E0D0', '#7FFFAA', '#00FA9A',
  // '#F5FFFA',  '#3CB371', '#2E8B57', '#F0FFF0',
  // '#98FB98', '#8FBC8F', '#32CD32', '#00FF00', '#228B22', '#008000',
  //  '#7CFC00', '#ADFF2F', '#556B2F',
  // '#FAFAD2', '#FFFFF0', '#FFFFE0', '#FFFF00', '#808000',
  // '#FFFACD', '#EEE8AA', '#F0E68C', '#FFD700', '#FFF8DC', '#DAA520',
  // '#FFFAF0', '#FDF5E6', '#F5DEB3', '#FFE4B5', '#FFA500', '#FFEFD5',
  // '#FFEBCD', '#FFDEAD', '#FAEBD7', '#D2B48C',
]
let useColorList = []

/**
 * 从ColorList不重样获取color
 * @returns {string}
 */
export function getRandomColor() {
  let number = 0;
  while (useColorList.findIndex(e => e === number) !== -1) {
    number = Math.random() * ColorList.length;
    number = Math.floor(number);
  }
  useColorList.push(number)
  if (useColorList.length >= ColorList.length / 2) {
    useColorList = []
  }
  return ColorList[number]
}


//脱敏
export const DesensitizationData = {
  maskChineseName(name) {
    if (!name) {
      return name
    }
    if (name.length === 2) {
      // 姓只显示第一个字符，名隐藏
      return name[0] + '*';
    } else if (name.length > 2) {
      // 姓显示第一个字符，名只显示第一个字符，其他字符隐藏
      return name[0] + '*' + name[name.length - 1];
    } else {
      return name; // 姓名长度不足，不脱敏
    }
  },
  maskChineseAddress(address) {
    if (!address) {
      return address
    }
    address=address.split('')
    // 保留省份、城市和区县信息，其余部分用星号(*)替代
    for (let i = 0; i < address.length; i++) {
      if (!isNaN(Number(address[i]))) {
        address[i]='*'
      }
    }
    return address.join('')
  }
}

//设置地图连接
export function setMapUrlByQuery(query){
  if (query&&typeof query==='object'){
    let key=AppBaseConfig.DigitalTwinCloudUrl
    localStorage.setItem(key,query[key])
  }
}
