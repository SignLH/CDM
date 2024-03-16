<template>
    <div class="my-pagination">
        <div class="dian dianL" v-if="currentPage > 3 && pages > 5" @click="delCurrentPage(0)">...</div>
        <div class="number">
            <span v-for="item in list" :key="item" :class="{ active: currentPage === item }"
                @click="changePage(item)">{{ item }}</span>
        </div>
        <div class="dian dianR" v-if="currentPage < pages - 2 && pages > 5" @click="delCurrentPage(1)">...</div>
    </div>
</template>
<script>
export default {
    props: {
        total: {
            type: Number,
            default: 80
        },
        currentPage: {
            type: Number,
            default: 1
        },
        pagesize: {
            type: Number,
            default: 10
        }
    },
    computed: {
        // 计算总页数
        pages() {
            return Math.ceil(this.total / this.pagesize)
        },
        // 页码显示组合
        list() {
            const result = []
            // 总页数小于等于5页的时候
            if (this.pages <= 5) {
                for (let i = 1; i <= this.pages; i++) {
                    result.push(i)
                }
            } else {
                // 总页数大于5页的时候
                // 控制两个极端那边的省略号的有无，页码的显示个数与选中页码居中
                if (this.currentPage <= 2) {
                    for (let i = 1; i <= 5; i++) {
                        result.push(i)
                    }
                } else if (this.currentPage >= 3 && this.currentPage <= this.pages - 2) {
                    for (let i = this.currentPage - 2; i <= this.currentPage + 2; i++) {
                        result.push(i)
                    }
                } else if (this.currentPage > this.pages - 2) {
                    for (let i = this.pages - 4; i <= this.pages; i++) {
                        result.push(i)
                    }
                }
            }
            return result
        }
    },
    data() {
        return {
        }
    },
    methods: {
        // 点击上一页下一页页码改变页码
        changePage(type) {
            console.log(type)
            // 点击上一页按钮
            let currentPage = this.currentPage
            if (type === false) {
                if (this.currentPage <= 1) return
                currentPage -= 1
            } else if (type === true) {
                // 点击下一页按钮
                if (this.currentPage >= this.pages) return
                currentPage += 1
            } else {
                // 点击页码
                currentPage = type
            }
            // 传给父组件当前页码，可以在该事件中做相关操作
            this.$emit('change', currentPage)
        },
        // 点击省略号前后位移三位
        delCurrentPage(type) {
            if (type) {
                const currentPage = this.currentPage + 3
                this.$emit('change', currentPage)
            } else {
                const currentPage = this.currentPage - 3
                this.$emit('change', currentPage)
            }
        }
    },
    mounted() {
    }
}
</script>
<style scoped lang="scss">
.my-pagination {
    display: flex;
    justify-content: center;
    align-items: center;
    position: relative;
    padding-top: 8px;
    width: 100%;

    .number {
        >span {
            font-family: 'arial';
            font-size: 20px;
            min-width: 30px;
            padding: 0 5px;
            height: 30px;
            display: inline-block;
            text-align: center;
            line-height: 30px;
            color: #ffffff;
            cursor: pointer;

            &.active {
                color: #14e1cd;
                font-size: 22px;
                font-weight: bold;
                text-decoration-line: underline;
            }
        }
    }

    .dian {
        width: 30px;
        height: 30px;
        color: #ffffff;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        line-height: 24px;
        cursor: pointer;
    }
}
</style>
  