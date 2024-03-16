const ColorList = [
  '#FFB6C1', '#FFC0CB', '#DC143C', '#FFF0F5', '#DB7093', '#FF69B4',
  '#FF1493', '#C71585', '#DA70D6', '#D8BFD8', '#DDA0DD', '#EE82EE',
  '#FF00FF', '#FF00FF', '#8B008B', '#800080', '#BA55D3', '#9400D3',
  '#9932CC', '#4B0082', '#8A2BE2', '#9370DB', '#7B68EE', '#6A5ACD',
  '#483D8B', '#E6E6FA', '#F8F8FF', '#0000FF', '#0000CD', '#191970',
  '#00008B', '#000080', '#4169E1', '#6495ED', '#B0C4DE', '#778899',
  '#708090', '#1E90FF', '#F0F8FF', '#4682B4', '#87CEFA', '#87CEEB',
  '#00BFFF', '#ADD8E6', '#B0E0E6', '#5F9EA0', '#F0FFFF', '#E1FFFF',
  '#AFEEEE', '#00FFFF', '#00FFFF', '#00CED1', '#2F4F4F', '#008B8B',
  '#008080', '#48D1CC', '#20B2AA', '#40E0D0', '#7FFFAA', '#00FA9A',
  '#F5FFFA', '#00FF7F', '#3CB371', '#2E8B57', '#F0FFF0', '#90EE90',
  '#98FB98', '#8FBC8F', '#32CD32', '#00FF00', '#228B22', '#008000',
  '#006400', '#7FFF00', '#7CFC00', '#ADFF2F', '#556B2F', '#6B8E23',
  '#FAFAD2', '#FFFFF0', '#FFFFE0', '#FFFF00', '#808000', '#BDB76B',
  '#FFFACD', '#EEE8AA', '#F0E68C', '#FFD700', '#FFF8DC', '#DAA520',
  '#FFFAF0', '#FDF5E6', '#F5DEB3', '#FFE4B5', '#FFA500', '#FFEFD5',
  '#FFEBCD', '#FFDEAD', '#FAEBD7', '#D2B48C',
]
let useColorList = []

export function getRandomColor() {
  let number = 0;
  while (useColorList.findIndex(e => e === number) !== -1) {
    number = Math.random() * ColorList.length;
    number = Math.round(number);
  }
  useColorList.push(number)
  if (useColorList.length >= ColorList.length / 2) {
    useColorList = []
  }
  return ColorList[number]
}

// for (let i = 0; i < 500; i++) {
//   console.log(getRandomColor())
// }



/**
 * 获取相近颜色方法
 * @param color hex
 * @param step <=1
 * @param isToWhite 整体颜色向方向
 * @returns {string} rgba
 */
export function getSimilarColor(color, step, isToWhite = true) {
  const direction = isToWhite ? 1 : -1
  let r = parseInt(color.slice(1, 3), 16) + step * 10 * direction;
  let g = parseInt(color.slice(3, 5), 16) + step * 10 * direction;
  let b = parseInt(color.slice(5, 7), 16) + step * 10 * direction;

  const checkValue = (v) => {
    if (v > 255) v = 255
    else if (v < 0) v = 0
    return v
  }
  r = checkValue(r)
  g = checkValue(g)
  b = checkValue(b)
  return "rgba(" + r + ", " + g + ", " + b + ", " + step + ")";
}


export function getRGBAColor(color, opacity) {
  let r = parseInt(color.slice(1, 3), 16) ;
  let g = parseInt(color.slice(3, 5), 16);
  let b = parseInt(color.slice(5, 7), 16) ;
  return "rgba(" + r + ", " + g + ", " + b + ", " + opacity + ")";
}


export const XjdwTypeColor=[
  {name:'社区志愿服务中心',color:'#FFB6C1'}, {name:'文明实践分中心',color:'#FF1493'}, {name:'文明实践站',color:'#FF00FF'},
  {name:'三中心',color:'#9932CC'}, {name:'学校',color:'#3CB371'}, {name:'交通路口',color:'#0edcdc'},
  {name:'主次干道/背街小巷',color:'#708090'}, {name:'超市',color:'#FFD700'}, {name:'邮政局',color:'#2E8B57'},
  {name:'银行',color:'#4B0082'}, {name:'电信',color:'#F0F8FF'}, {name:'车站',color:'#D2B48C'},
  {name:'乡镇',color:'#FFFF00'}, {name:'集贸市场及周边',color:'#DA70D6'}, {name:'避难场所',color:'#EEE8AA'},
  {name:'社区（小区）',color:'#AFEEEE'},
]
