
/**
 * 根据 比对key 返回对象特定的 value
 * @param {*} list 源数据
 * @param {*} compareKey 对比key
 * @param {*} compareValue 对比值
 * @param {*} resKey 返回的key
 */
export function getListValueByKey(list, compareKey, compareValue, resKey) {
    let resValue = ""

    list.forEach(i => {
        if (i[compareKey] == compareValue) {
            resValue = i[resKey]
        }
        if (resValue == '' && i.children && i.children.length > 0) {
            resValue = getListValueByKey(i.children, compareKey, compareValue, resKey)
        }
    });

    return resValue
}

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


/**
 * This is just a simple version of deep copy
 * Has a lot of edge cases bug
 * If you want to use a perfect deep copy, use lodash's _.cloneDeep
 * @param {Object} source
 * @returns {Object}
 */
export function deepClone(source) {
    if (!source && typeof source !== 'object') {
        throw new Error('error arguments', 'deepClone')
    }
    const targetObj = source.constructor === Array ? [] : {}
    Object.keys(source).forEach(keys => {
        if (source[keys] && typeof source[keys] === 'object') {
            targetObj[keys] = deepClone(source[keys])
        } else {
            targetObj[keys] = source[keys]
        }
    })
    return targetObj
}

//【本项目特有】 -- 根据 源数据列表里的 plate sub_plate key_name 返回 value
export function formatPageDataByKeys(list, plate, sub_plate, key_name) {
    let value = ''
    list.forEach(i => {
        if (i.plate == plate && i.sub_plate == sub_plate && i.key_name == key_name) {
            value = i.value
        }
    })
    return value
}