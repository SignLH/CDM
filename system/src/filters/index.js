//时间格式化
export function formatDate(date, formatStr) {
    if (date === null) {
        return ''
    }
    date = new Date(date);

    let str = formatStr;
    let Week = ['日', '一', '二', '三', '四', '五', '六'];
    str = str.replace(/yyyy|YYYY/, date.getFullYear());
    str = str.replace(/MM/, (date.getMonth() + 1) > 9 ? (date.getMonth() + 1).toString() : '0' + (date.getMonth() + 1));
    str = str.replace(/w|W/g, Week[date.getDay()]);
    str = str.replace(/dd|DD/, date.getDate() > 9 ? date.getDate().toString() : '0' + date.getDate());
    str = str.replace(/hh|HH/, date.getHours() > 9 ? date.getHours().toString() : '0' + date.getHours());
    str = str.replace(/mm/, date.getMinutes() > 9 ? date.getMinutes().toString() : '0' + date.getMinutes());
    str = str.replace(/ss|SS/, date.getSeconds() > 9 ? date.getSeconds().toString() : '0' + date.getSeconds())
    return str
}

//信息格式化
export function formatStringInfo(str) {
    let res = ''
    if (str == null || str == '') {
        res = 'n***l'
    } else if (str.length > 4) {
        res =
            str.substring(0, 3) +
            '****' +
            str.substring(str.length - 4, str.length - 1)
    } else {
        res = str.substring(0, 1) +
            '****' +
            str.substring(str.length - 1, str.length - 1)
    }
    return res
}

//百分比格式化
export function formatPercentage(value) {
    if (value === null || value === '') {
        return ''
    } else {
        return (Number(value) * 100).toFixed(2) + '%'
    }
}


//【本项目特有】 -- 根据 源数据列表里的 plate sub_plate key_name 返回 value
export function formatPageDataByKeys(list, plate, sub_plate, key_name) {
    let value = '0'
    list.forEach(i => {
        if (i.plate == plate && i.sub_plate == sub_plate && i.key_name == key_name) {
            value = i.value
        }
    })
    return value
}

//【本项目特有】 -- 根据数值大小获取状态码 有用于 状态渲染
export function formatStatusByValue(val) {
    let status = 'normal'
    
    const level1 = ['正常','优']
    const level2 = ['良']

    if (level1.includes(val) || Number(val) >= 85) {
        status = 'success'
    } else if (level2.includes(val) || Number(val) >= 60) {
        status = 'warning'
    } else {
        status = 'error'
    }
    return status
}