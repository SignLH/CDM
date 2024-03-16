(function () {
    var multiHelper = null;

    var videoParams = {
        desDiv: null,
        desSession: null,
        szSzChanelInfo: [],
        szRcSzChannel: [],
        szDesDiv: [],
        szDialog: [],
        closeImgBase64: null,
        maxImgBase64: null,
        iLastCount: 0
    };



    function nessInfoProxy(playPanelData, handle, scope) {
        var channelInfo = playPanelData.getChannelInfo();
        var chanel = videoParams.desSession.swGetPuChanel(channelInfo.puId, channelInfo.index);
        handle.bind(scope)(chanel);
    }

    /**
     * play panel data , to record play div info
     * @param {Document} contentDiv 
     * @param {Document} OptionsDiv 
     * @param {Number} index 
     */
    function PlayPanelData(contentDiv, OptionsDiv, index, recordFlagDiv) {
        this.content = contentDiv;
        this.recordFlagDiv = recordFlagDiv;
        // this.options = OptionsDiv;
        this._bPlaying = false;
        this._index = index;
        this._hDlg = 0;
        this._busy = false;
        this._channelInfo = {
            puId: "",
            index: -1,
            iPlayId: -1
        };
        this._hDlg = -1;
        this._callbackInfo = {
            szPlayData: null,
            callback: null,
            tag: null
        },
        this._localRecordPlayInfo = {
            bStart: false,
            onResult: null,
            userData: null
        };
        this.CheckBFocus = function(){
            return CheckDivHasBeenFocused(this._index);
        }
    }

    PlayPanelData.prototype = {
        setLocalResultInfo: function(onResult, userData){
            this._localRecordPlayInfo.onResult = onResult;
            this._localRecordPlayInfo.userData = userData;
        },
        getLocalResultInfo: function(){
            return this._localRecordPlayInfo;
        },
        setLocalRecordStatus: function(bStart){
            this._localRecordPlayInfo.bStart = bStart;
            if(bStart){
                this.recordFlagDiv.style.display = "block";
            }else{
                this.recordFlagDiv.style.display = "none";
            }
        },
        getLocalRecordStatus: function(){
            var bLocalRecord = this.recordFlagDiv.style.display == "block";
            return bLocalRecord;
        },
        bPlaying: function () {
            return this._bPlaying;
        },
        setPlay: function () {
            this._bPlaying = true;
        },
        setClose: function () {
            this._bPlaying = false;
        },
        updateIndex: function (index) {
            this._index = index;
        },
        setHDlg: function (hDlg) {
            this._hDlg = hDlg;
        },
        free: function () {
            this.delDivUserData();
        },
        bFree: function () {
            return !this._bPlaying && !this._busy;
        },
        hidePlay: function () {
            nessInfoProxy(this, function (channel) {
                channel.swHide();
            }, this);
        },
        showPlay: function () {
            nessInfoProxy(this, function (channel) {
                channel.swShow();
            }, this);
        },
        setChannelData: function (puId, index, iPlayId) {
            this._channelInfo.puId = puId;
            this._channelInfo.index = index;
            this._channelInfo.iPlayId = iPlayId;
            this.setDivUserData();
        },
        getChannelInfo: function () {
            return this._channelInfo;
        },
        getDivUserData: function () {
            return this.content.dataDivTag;
        },
        setDivUserData: function () {
            this.content.dataDivTag = this;
        },
        delDivUserData: function () {
            this.content.dataDivTag = null;
        },
        getContentDiv: function () {
            return this.content;
        },
        getRecordFlag: function(){
            return this.recordFlagDiv;
        },
        // getOptionsDiv: function () {
        //     return this.options;
        // },
        beginOpen: function () {
            this._busy = true;
        },
        onOpenHasResult: function (bResult, hDlg) {
            if (bResult) {
                this._bPlaying = true;
                this._hDlg = hDlg;
            } else {
                this._bPlaying = false;
            }
            this._busy = false;
            this.checkAndEmitCallback();
            if (!bResult) {
                this.onCloseChanelClean();
            }
        },
        onCloseChanelClean: function () {
            // this.options.innerHTML = "";
            this._channelInfo = {
                puId: "",
                index: -1,
                iPlayId: -1
            };
            this._bPlaying = false;
            this._busy = false;
            this.setLocalRecordStatus(false);
        },
        getPlayDivInfo: function () {
            var curPlayInfo = {
                playid: this._index,
                bplaying: this._bPlaying,
                puid: this._channelInfo.puId,
                index: this._channelInfo.index
            };
            return curPlayInfo;
        },
        setCallback: function (szDesDiv, callback, tag) {
            this._callbackInfo.szPlayData = szDesDiv;
            this._callbackInfo.callback = callback;
            this._callbackInfo.tag = tag;
        },
        checkOpenHasResult: function () {
            return !this._busy;
        },
        checkAndEmitCallback: function () {
            if (this._callbackInfo.szPlayData) {
                var bAllHasResult = true;
                for (var iIndex in this._callbackInfo.szPlayData) {
                    if (!this._callbackInfo.szPlayData[iIndex].checkOpenHasResult()) {
                        bAllHasResult = false;
                        break;
                    }
                }
                if (bAllHasResult) {
                    var szResult = gPlayDivMgr.getPlayResult(this._callbackInfo.szPlayData);
                    this._callbackInfo.callback(szResult, this._callbackInfo.tag);
                }
            }

            this._callbackInfo.szPlayData = null;
            this._callbackInfo.callback = null;
            this._callbackInfo.tag = null;
        },
        beforeFrame: function () {
            if (this._bPlaying) {
                this.hidePlay();
            }
        }
    };

    function PlayDivMgr() {
        this._szPlayDivData = [];
        this._bInit = false;
    }

    PlayDivMgr.prototype = {
        Init: function () {
            if (!this._bInit) {
                this._bInit = true;
                for (var iIndex = 0; iIndex < 16; iIndex++) {
                    this.generatePlayPanel(null, iIndex);
                }
            }
        },
        getOneFreeDiv: function (iPlayId, puId, index) {
            var desDivDataTemp = null;
            if (typeof iPlayId == Number) {
                desDivDataTemp = this._CheckAndCloseChannel(iPlayId);
            } else {
                desDivDataTemp = this._GetOneFreeData();
            }
            if (desDivDataTemp) {
                desDivDataTemp.setChannelData(puId, index, iPlayId);
            }
            return desDivDataTemp;
        },
        _CheckAndCloseChannel: function (index) {
            var aimData = this._szPlayDivData[index];
            if (aimData && aimData.bPlaying()) {
                closeAChanelByUserData(aimData);
                aimData.free();
            }
            return aimData;
        },
        _GetOneFreeData: function () {
            var tempPlayDivData = null;
            for (var iIndex in this._szPlayDivData) {
                tempPlayDivData = this._szPlayDivData[iIndex];
                if (tempPlayDivData.bFree()) {
                    return tempPlayDivData;
                }
            }
            return null;
        },
        getPlayDataById: function (iPlayId) {
            return this._szPlayDivData[iPlayId];
        },
        generatePlayPanel: function (parentDiv, iPlayId) {
            var playContentDiv = null;
            var playOptionsDiv = null;
            var recordFlagDiv = null;
            var desPanelData = this.getPlayDataById(iPlayId);
            if (desPanelData) {
                playContentDiv = desPanelData.getContentDiv();
                // playOptionsDiv = desPanelData.getOptionsDiv();
            } else {
                playContentDiv = createEleAddClassName('div', "playContentDiv_" + iPlayId);
                // playOptionsDiv = creatEleAddClassName('div', "playOptionsDiv_" + iPlayId);
                playContentDiv.style.backgroundColor = "black";
                playContentDiv.style.border = "1px solid black";
                playContentDiv.style.height = "100%";
                playContentDiv.style.width = "100%";
                playContentDiv.onclick = onPlayDivClick;
                playContentDiv.style.position = 'relative';

                recordFlagDiv = createEleAddClassName('div', "recordFlagPanel_" + iPlayId);
                recordFlagDiv.style.width = "100%";
                recordFlagDiv.style.height = "100%";
                recordFlagDiv.style.zIndex = 1;
                recordFlagDiv.style.position = 'absolute';
                recordFlagDiv.ondblclick = onPlayDivDbClick;

                playContentDiv.appendChild(recordFlagDiv);

                var recordFlag = createEleAddClassName('div', "recordFlag_" + iPlayId);
                recordFlag.style.width = "15px";
                recordFlag.style.height = "15px";
                recordFlag.style.display = "none";
                recordFlag.style.position = 'absolute';
                recordFlag.style.right = "10px";
                recordFlag.style.bottom = "10px";
                recordFlag.style.backgroundColor = "red";
                recordFlag.style.borderRadius = "15px";
                recordFlagDiv.appendChild(recordFlag);

                desPanelData = new PlayPanelData(playContentDiv, playOptionsDiv, iPlayId, recordFlag);
                this._szPlayDivData.push(desPanelData);
            }

            if (desPanelData.bPlaying()) {
                desPanelData.hidePlay();
            }

            // playOptionsDiv.style.backgroundColor = "#ccc";
            if (parentDiv) {
                parentDiv.appendChild(playContentDiv);
                // parentDiv.appendChild(playOptionsDiv);
            }

            if (desPanelData.bPlaying()) {
                desPanelData.showPlay();
            }

            return desPanelData;
        },
        getDivUserDataByPlayId: function (iPlayId) {
            return this._szPlayDivData[iPlayId];
        },
        getPlayInfo: function (iPmCount) {
            var playInfoDates = [];

            var tempPlayData = null;
            for (var iIndex = 0; iIndex < iPmCount; iIndex++) {
                tempPlayData = this.getDivUserDataByPlayId(iIndex);
                playInfoDates.push(tempPlayData.getPlayDivInfo());
            }

            return playInfoDates;
        },
        /**
         * 
         * @param {Array} szPlayDivData 
         */
        getPlayResult: function (szPlayDivData) {
            var szResult = [];
            var playDivData = null;
            for (var iIndex in szPlayDivData) {
                playDivData = szPlayDivData[iIndex];
                szResult.push(playDivData.getPlayDivInfo());
            }
            return szResult;
        },
        beforeFrame: function () {
            var tempPlayDivData;
            for (var iIndex in this._szPlayDivData) {
                tempPlayDivData = this._szPlayDivData[iIndex];
                if (tempPlayDivData.bPlaying()) {
                    tempPlayDivData.beforeFrame();
                }
            }
        },
        /**
         * 
         * @param {Function} handle 
         * @param {*} scope 
         */
        each: function (handle, scope) {
            var tempPlayDivData;
            for (var iIndex in this._szPlayDivData) {
                tempPlayDivData = this._szPlayDivData[iIndex];
                if(handle.bind(scope)(tempPlayDivData)){
                    break;
                }
            }
        }
    };


    var gPlayDivMgr = new PlayDivMgr();

    var _eventHandle = {
        _handler: null,
        _emit: function (events, args) {
            if (this._handler && typeof this._handler == "function") {
                this._handler(events, args);
            } else {
                multiHelper.log("glb event hander not set!");
            }
        },
        _events: {
            selectScreen: "selectScreen"
        },
        setHandler: function (hander) {
            this._handler = hander;
        },
        selectDivEmit: function (iPlayId) {
            var resultData = null;
            var playData = gPlayDivMgr.getPlayDataById(iPlayId);
            if (playData) {
                resultData = playData.getPlayDivInfo();
            } else {
                resultData = null;
            }

            this._emit(this._events.selectScreen, resultData);
        }
    };


    var modelDiv = {
        mainContainer: createEleAddClassName("div", "mpMainContainer"),
        szItemDiv: [],
        addItemDiv: function (div) {
            this.szItemDiv.push(div);
        },
        getItemDiv: function (iIndex) {
            var desItemDiv = this.szItemDiv[iIndex];
            if (!desItemDiv) {
                desItemDiv = createEleAddClassName('div', "displayItemDiv" + iIndex);
                // desItemDiv.ondblclick = onPlayDivdblClick;
                this.addItemDiv(desItemDiv);
            }
            return desItemDiv;
        },
        szSubItemDiv: [],
        getSubItem: function (iIndex) {
            var desItemDiv = this.szSubItemDiv[iIndex];
            if (!desItemDiv) {
                desItemDiv = createEleAddClassName('div', "subItemDiv");
                this.szSubItemDiv.push(desItemDiv);
            }
            return desItemDiv;
        }
    }
    modelDiv.mainContainer.style.background = "white";

    // function onPlayDivdblClick (event){
    //     var hgt = $(event.target).height;
    //     var wdh = $(event.target).width;
    //     var phgt = $(event.target).parent().height;
    //     var pwdh = $(event.target).parent().width;
    //     if(hgt < phgt || wdh < pwdh){
    //         hgt = phgt;
    //         wdh = pwdh;
    //         console.log("55555")
    //     }
    // }

    /**
     * generate multi screen
     * @param {destination div} desDiv 
     * @param {*} iCount 
     */
    function elementsInit(desDiv, iCount) {
        var mainContainer = modelDiv.mainContainer;
        mainContainer.style.height = "100%";
        mainContainer.style.width = "100%";
        mainContainer.style.paddingRight = "0x";
        mainContainer.style.boxSizing = "border-box";
        mainContainer.style.position = "relative";
        desDiv.appendChild(mainContainer);
        var iDim = calculateDim(iCount);
        var itemDiv = null;
        var subItemDiv = null;
        var stepPercent = (100 / iDim) + "%";
        iDim = iDim * iDim;

        for (var iIndex = 0; iIndex < 16; iIndex++) {
            itemDiv = modelDiv.getItemDiv(iIndex);
            mainContainer.appendChild(itemDiv);
            if (iIndex < iDim) {
                itemDiv.style.height = stepPercent;
                itemDiv.style.width = stepPercent;
                itemDiv.style.display = "block";
            } else {
                itemDiv.style.display = "none";
            }
            itemDiv.style.float = "left";
            itemDiv.style.padding = "1.5px";
            itemDiv.style.boxSizing = "border-box";

            subItemDiv = modelDiv.getSubItem(iIndex);
            subItemDiv.style.border = "1px solid black";
            subItemDiv.style.height = "100%";
            subItemDiv.style.width = "100%";
            subItemDiv.style.boxSizing = "border-box";
            subItemDiv.style.cursor = "pointer";
            // subItemDiv.onclick = onPlayDivClick;
            itemDiv.appendChild(subItemDiv);
            gPlayDivMgr.generatePlayPanel(subItemDiv, iIndex);
        }
        return iDim;
    }

    var lastFocusDiv = null,
        bLockCssChange = false;

    function onPlayDivClick(event) {
        if (!bLockOptionsHandleClick) {
            clearPlayDivShown();
            fillPlayDivShown(this);
            bLockCssChange = true;
        } else {
            bLockOptionsHandleClick = false;
            bLockCssChange = true;
        }
    }

    function onPlayDivDbClick(event){
        beforeFSFlagInfo(event);
    }

    var bFullScreen = false;

    function beforeFSFlagInfo(event) {
        var aimDom = event.target.parentNode;
        CheckAndFullScreen(aimDom);
    }

    function onPlayDivDbClick(event) {
        var aimDom = event.target.parentNode;
          launchIntoFullscreen(modelDiv.mainContainer);
    }


    function rememberFulldiv(aimDom) {
        var aimDomdisitem = aimDom.parentNode.parentNode;
        fulldivused.height = aimDomdisitem.currentStyle.height;
        fulldivused.width = aimDomdisitem.currentStyle.width;
    }

    var fullparentscreen = false;
    var fulldivused = {};

    function CheckAndFullScreen(aimDom) {
        if (aimDom && aimDom.dataDivTag) {
            var channelInfo = aimDom.dataDivTag._channelInfo;
            var chanel = videoParams.desSession.swGetPuChanel(channelInfo.puId, channelInfo.index);
            var aimDomdisitem = aimDom.parentNode.parentNode;
            var mapaimDom = aimDomdisitem.parentNode;
            if (!fullparentscreen) {
                chanel.swHide();
                rememberFulldiv(aimDom);
                aimDomdisitem.style.position = "absolute"
                aimDomdisitem.style.height = "100%";
                aimDomdisitem.style.width = "100%";
                aimDomdisitem.style.zIndex = "100";

                mapaimDom.style.overflow = "hidden";
                fullparentscreen = true;
                chanel.swShow();
            } else {
                aimDomdisitem.style.height = fulldivused.height;
                aimDomdisitem.style.width = fulldivused.width;

                aimDomdisitem.style.position = "relative";
                mapaimDom.style.overflow = "visible";
                aimDomdisitem.style.zIndex = "0";
                fullparentscreen = false;
            }
        }
    }


    function CheckDivHasBeenFocused(id){
        if(lastFocusDiv == null){
            return false;
        }
        var fId = lastFocusDiv.className.replace("playContentDiv_", "");
        return id == Number(fId);
    }

    function fillPlayDivShown(focusEle) {
        focusEle.style.borderColor = "red";
        lastFocusDiv = focusEle;
        
        if(lastFocusDiv){
            var playData = gPlayDivMgr.getPlayDataById(lastFocusDiv.className.replace("playContentDiv_", ""));
            DoAudioMediaChange(playData);
        }
        var videoDiv = $(focusEle).parent()
        var iIndex = $(".subItemDiv").index(videoDiv);
        _eventHandle.selectDivEmit(iIndex);
    }

    function clearPlayDivShown() {
        if (lastFocusDiv) {
            lastFocusDiv.style.borderColor = "black";
            $(lastFocusDiv).css("background", "black");
        }

        if(lastFocusDiv){
            var tempLastDiv = lastFocusDiv;
            lastFocusDiv = null;

            var playData = gPlayDivMgr.getPlayDataById(tempLastDiv.className.replace("playContentDiv_", ""));
            DoAudioMediaChange(playData);
        }
    }

    var isClearEmit = false;
    // $(document).click(function () {
    //     if (bLockCssChange) {
    //         bLockCssChange = false;
    //         isClearEmit = true;
    //     } else if (isClearEmit) {
    //         isClearEmit = false;
    //         clearPlayDivShown();
    //         _eventHandle.selectDivEmit(-1);
    //     }
    // });

    function calculateDim(iCount) {
        var tempCount = 0;
        var tempCountD = 0;
        for (var ii = 1; ii < 5; ii++) {
            tempCount = ii * ii;
            tempCountD = (ii - 1) * (ii - 1);
            if (iCount >= tempCountD && iCount <= tempCount) {
                return ii;
            }
        }
        return -1;
    }

    /**
     * 判断输入屏幕数
     * @param {Number} iCount 
     * @returns {boolean}
     */
    function judgeFrameCount(iCount) {
        var bLegal = iCount >= 1 && iCount <= 16;
        if (!bLegal) {
            multiHelper.log("Frame Count is invalid");
        }
        return bLegal;
    }


    function onClickCloseAllScreen() {
        closeAllScreen(false);
    }

    /*
     *清除所有播放窗口
     */
    function closeAllScreen(bContinueDialog) {
        var bCloseAll = typeof bContinueDialog == "undefined" || !bContinueDialog;
        if (bCloseAll) {
            for (var dialogIndex in videoParams.szDialog) {
                closeAChanelByUserData(videoParams.szDialog[dialogIndex]);
            }
        }
    }

    function openChannels(szSzChanelInfo, callback, tag) {
        var tempPu = null;
        var playDivData = null;
        var playDivDataUses = [];
        for (var iIndex = szSzChanelInfo.length - 1; iIndex >= 0; iIndex--) {
            playDivData = gPlayDivMgr.getOneFreeDiv(szSzChanelInfo[iIndex].playid, szSzChanelInfo[iIndex].puid, szSzChanelInfo[iIndex].index);
            playDivDataUses.push(playDivData);
            playDivData.beginOpen();
        }

        var bAllFailed = true;
        for (var iIndex = 0; iIndex < playDivDataUses.length; iIndex++) {
            playDivData = playDivDataUses[iIndex];
            if (playDivData) {
                if (openChannel(playDivData)) {
                    bAllFailed = false;
                    playDivData.setCallback(playDivDataUses, callback, tag);
                }
            }
        }

        if (bAllFailed) {
            var szResult = gPlayDivMgr.getPlayResult(playDivDataUses);
            callback(szResult, tag);
        }
    }


    function openChannel(playDivData) {
        var channelInfo = playDivData.getChannelInfo();
        var div = playDivData.getContentDiv();
        multiHelper.log("getChanelFromJsw:" + channelInfo.puId + channelInfo.index);
        var chanel = videoParams.desSession.swGetPuChanel(channelInfo.puId, channelInfo.index);
        if (chanel) {
            var iAudio = playDivData.CheckBFocus() ? jSW.MEDIADIR.AUDIORECV : 0;
			console.log(jSW.MEDIADIR.VIDEORECV+"+++"+iAudio);
					console.log(jSW.MEDIADIR.VIDEORECV | iAudio);
            var rc = chanel.swOpenEx({
                div: div,
                media: 10,
                callback: onOpenChannelHasResult,
                bstretch: true,
                bwindowless: true
            });
            if (rc != jSW.RcCode.RC_CODE_S_OK) {
                multiHelper.log(channelInfo.puId + channelInfo.index + "open faild");
                playDivData.onOpenHasResult(false);
                return false;
            }
        } else {
            multiHelper.log(channelInfo.puId + channelInfo.index + "not found");
            playDivData.onOpenHasResult(false);
            return false;
        }
        return true;
    }

    function DoAudioMediaChange(playDivData){
        if(playDivData._bPlaying){
            var channelInfo = playDivData.getChannelInfo();
            var div = playDivData.getContentDiv();
            multiHelper.log("getChanelFromJsw:" + channelInfo.puId + channelInfo.index);
            if(videoParams.desSession){
                var chanel = videoParams.desSession.swGetPuChanel(channelInfo.puId, channelInfo.index);
                if(chanel){
                    var bRecvAudio = playDivData.CheckBFocus();
                    chanel.swAlterVideoDir({
                        hdlg: playDivData._hDlg,
                        isRecvAudio: bRecvAudio,
                        callback: onAlterVideoDirResult
                    });
                }
            }
        }
    }

    function onAlterVideoDirResult(opt, response){
        
    }

    function onOpenChannelHasResult(opt, response, data) {
        var bResult = response.emms.code == jSW.RcCode.RC_CODE_S_OK;
        var playDivData = opt.div.dataDivTag;
        var hDlg = 0;
        if (data) {
            hDlg = data.hdlg;
        }
        if (bResult) {
            onOpenChanelOk(playDivData);
        }
        playDivData.onOpenHasResult(bResult, hDlg);
    }


    function onOpenChanelOk(playDivData) {
        videoParams.szDialog.push(playDivData);
        if (playDivData) {
            onOpenChanelOkAddHandle(playDivData);
        }
    }

    function onOpenChanelOkAddHandle(playDivData) {
        var handleClose = createEleAddClassName("img", "pmHandleClose");
        var handleMax = createEleAddClassName("img", "pmHandleMax");
        handleClose.src = videoParams.closeImgBase64;
        handleMax.src = videoParams.maxImgBase64;
        handleMax.style.height = "12px";
        handleMax.style.width = "15px";
        handleMax.style.cursor = "pointer";
        handleMax.style.marginRight = "5px";
        handleClose.style.height = "20px";
        handleClose.style.width = "20px";
        handleClose.style.marginRight = "5px";
        handleClose.style.cursor = "pointer";
        handleClose.onclick = onCloseHandleClick;
        handleMax.onclick = onMaxHandleClick;
        handleMax.style.float = "left";
        handleMax.style.marginLeft = "10px";
        handleMax.style.marginTop = "3px";
    }

    function createEleAddClassName(eleName, eleClassName) {
        var focusEle = document.createElement(eleName);
        focusEle.className = eleClassName;
        return focusEle;
    }

    var bLockOptionsHandleClick = false;

    function onCloseHandleClick() {
        bLockOptionsHandleClick = true;
        var iPlayId = $(".subItemDiv").index($(this).parent().parent());
        closePreviewByPlayId(iPlayId);
    }


    function getLocalRecordInfoCb(options, response, data){
        var tagCb = options.tag.cb;
        var playId = options.tag.playId;
        tagCb(response.emms.code, playId, data);
    }

    function getLocalRecordInfoByPlayId(iPlayId, onGetLocalRecordInfo){
        var stRc = {
            rc: jSW.RcCode.RC_CODE_E_NOTFOUND,
            bRecord: false
        };
        var rc = jSW.RcCode.RC_CODE_E_NOTFOUND;
        var playDivData = gPlayDivMgr.getDivUserDataByPlayId(iPlayId);
        if(playDivData == null){
            return stRc;
        }
        var playInfo = playDivData.getPlayDivInfo();
        if(playInfo == null || (playInfo.bplaying == false)){
            return stRc;
        }
        stRc.rc = jSW.RcCode.RC_CODE_S_OK;
        stRc.bRecord = playDivData.getLocalRecordStatus();
        return stRc;
    }

    function onSetLocalRecordResult(options, response){
        var tagCb = options.tag.cb;
        var userData = options.tag.userData;
        var iPlayId = options.tag.playId;
        var bResult = response.emms.code == jSW.RcCode.RC_CODE_S_OK;
        var bStart = options.tag.bStart;
        if(bResult){
            var playDivData = gPlayDivMgr.getDivUserDataByPlayId(iPlayId);
            playDivData.setLocalRecordStatus(bStart);
        }
        multiHelper.log("record set has callback: " + response.emms.code);
        tagCb(iPlayId, response.emms.code, userData);
    }

    function onLocalRecordHasResult(stResult){
        var channel = stResult.channel;
        var storage = stResult.storage;
        var DstPlayId = -1; 
        var DstLRPlayInfo = null;
        var DstPlayDivData = null;
        gPlayDivMgr.each(function(PlayDivData){
            var playInfo = PlayDivData.getPlayDivInfo();
            if(playInfo.puid == channel._parent._id_pu && playInfo.index == channel._info_chanel.id){
                DstLRPlayInfo = PlayDivData.getLocalResultInfo();
                DstPlayId = playInfo.playid;
                DstPlayDivData = PlayDivData;
                return true;
            }
        }, this);
        if(DstPlayId < 0){
            return;
        }
        multiHelper.log("Local Record over: " + DstPlayId, storage);

        DstPlayDivData.setLocalRecordStatus(false);

        if(DstLRPlayInfo){
            var GRICb = DstLRPlayInfo.onResult;
            var userData = DstLRPlayInfo.userData;
            GRICb(DstPlayId, storage, userData);
        }
        
    }

    function localRecordOp(iPlayId, iRecordTimes, recordPath, rcCallback, recordCallback, userData) {
        var rc = jSW.RcCode.RC_CODE_E_NOTFOUND;
        if(iRecordTimes > 300 || iRecordTimes <= 0){
            var playDivData = gPlayDivMgr.getDivUserDataByPlayId(iPlayId);
            if(playDivData == null){
                return rc;
            }else{
                if(playDivData){
                    playDivData.setLocalResultInfo(recordCallback, userData);
                    var playInfo = playDivData.getPlayDivInfo();
                    var chanel = videoParams.desSession.swGetPuChanel(playInfo.puid, playInfo.index);
                    if(chanel){
                        var params = {
                            szpath: recordPath,
                            ifileleninseconds: iRecordTimes,
                            callback: onSetLocalRecordResult,
                            recordCallback: onLocalRecordHasResult,
                            tag: {
                                cb: rcCallback,
                                playId: iPlayId,
                                userData: userData,
                                bStart: iRecordTimes > 0
                            }
                        };
                        var rc = chanel.swSetLocalRecord(params);
                        multiHelper.log("record set has result" + rc);
                    }
                }
            }
        }else{
            multiHelper.log("hi, record seconds must above 300 due to nru limit");
        }
        return rc;
    }

    function closePreviewByPlayId(iPlayId) {
        if (iPlayId == -1) {
            gPlayDivMgr.each(function (data) {
                if (data.bPlaying()) {
                    closeAChanelByUserData(data);
                }
            }, this);
            return true;
        }

        var playDivData = gPlayDivMgr.getDivUserDataByPlayId(iPlayId);
        if (playDivData) {
            multiHelper.log("close preview" + iPlayId + "failed");
            return closeAChanelByUserData(playDivData);
        } else {
            return false;
        }
    }

    function closeAChanelByUserData(playDivData) {
        var channelInfo = playDivData.getChannelInfo();
        var rc = closeChannel(channelInfo.puId, channelInfo.index)
        playDivData.onCloseChanelClean();
        return rc;
    }

    function onMaxHandleClick() {
        bLockOptionsHandleClick = true;
        var iPlayId = $(".subItemDiv").index($(this).parent().parent());
        fullChannelByPlayId(iPlayId);
    }

    function fullChannelByPlayId(iPlayId) {
        var playDivData = gPlayDivMgr.getDivUserDataByPlayId(iPlayId);
        return fullChanel(playDivData);
    }

    function fullChanel(playDivData) {
        var channelInfo = playDivData.getChannelInfo();
        var chanel = videoParams.desSession.swGetPuChanel(channelInfo.puId, channelInfo.index);
        if (chanel) {
            chanel.swFullScreen();
        }
    }

    function closeChannel(puid, index) {
        var chanel = videoParams.desSession.swGetPuChanel(puid, index);
        if (chanel) {
            var rc = chanel.swClose();
            return rc == jSW.RcCode.RC_CODE_S_OK;
        } else {
            return false;
        }
    }

    function launchIntoFullscreen(element) {
        if (element.requestFullscreen) {
            element.requestFullscreen();
        } else if (element.mozRequestFullScreen) {
            element.mozRequestFullScreen();
        } else if (element.webkitRequestFullscreen) {
            element.webkitRequestFullscreen();
        } else if (element.msRequestFullscreen) {
            element.msRequestFullscreen();
        }
        // element.style.width = window.screen.width + "px";
        // element.style.height = window.screen.height + "px";
    }

    var divicount = null;

    function mpInit() {
        if (jSW.MP) {
            if (multiHelper) {
                multiHelper.log("Load Init Error, Load Init Cancel");
            }
            return;
        }
        jSW.MP = {
            /**
             * 
             */
            Init: function (desDiv, eventHandler) {
                if (jSW) {
                    if (jSW._Plugin.ComponentHelper) {
                        multiHelper = new jSW._Plugin.ComponentHelper({
                            name: "MultiScreenPreview"
                        });
                    }
                }
                gPlayDivMgr.Init();
                closeAllScreen(true);
                videoParams.desDiv = desDiv;
                _eventHandle.setHandler(eventHandler);
                multiHelper.log("Init Success");
                return jSW.RcCode.RC_CODE_S_OK;
            },

            /**
             * Change Screen Count
             * @param {Number} iCount 
             */
            Frame: function (iCount) {
                if (judgeFrameCount(iCount)) {
                    divicount = iCount;
                    gPlayDivMgr.beforeFrame();
                    var iPmCount = elementsInit(videoParams.desDiv, iCount);
                    videoParams.iLastCount = iPmCount;
                    var playInfos = gPlayDivMgr.getPlayInfo(iPmCount);
                    //multiHelper.log("Frames: " + playInfos.length + ", Call Frame Return Play Infos", playInfos);
                    return playInfos;
                }
                return null;
            },

            /**
				szSzPuid: [设备ID]
			*/
            Preview: function (session, szSzPuid, callback, tag) {
                if (!videoParams.desDiv) {
                    multiHelper.log("Display Div Is Not Set, Please Call Frame Before");
                    return;
                } else if (!session) {
                    multiHelper.log("Init Args Error");
                    return;
                } else if (szSzPuid.length == 0 || szSzPuid.length > 16) {
                    multiHelper.log("args 3, item count is zero to more than 16");
                } else {
                    videoParams.desSession = session;
                } {
                    openChannels(szSzPuid, callback, tag);
                }
            },

            PlayContainerFullScreen: function () {
                launchIntoFullscreen(modelDiv.mainContainer);
                return jSW.RcCode.RC_CODE_S_OK;
            },

            FullScreen: function (playId) {
                return fullChannelByPlayId(playId);
            },

            Close: function (playId) {
                return closePreviewByPlayId(playId);
            },

            GetStatus: function () {
                var playInfos = gPlayDivMgr.getPlayInfo(videoParams.iLastCount);
                return playInfos;
            },


            /**
             * 
             * @param {Number} iPlayId 播放ID
             * @param {Number} iRecordTimes 0: 停止录像  >0: 录像
             * @param {String} recordPath 本地录像路径 
             * @param {Function} rcCallback 录像接口调用回调
             * @param {Function} recordCallback 录像结果回调
             * @param {Object} userData 用户数据
             */
            LocalRecord: function (iPlayId, iRecordTimes, recordPath, rcCallback, recordCallback, userData) {
                return localRecordOp(iPlayId, iRecordTimes, recordPath, rcCallback, recordCallback, userData);
            },

            GetLocalRecordStatus: function(iPlayId){
                var stRc = getLocalRecordInfoByPlayId(iPlayId);
                return stRc;
            }
        }
    }
    if (typeof jSW == "undefined") {
        if (multiHelper) {
            multiHelper.log("Load Failed! jSW is undefined, Please load jSW.js before load this");
        }
    } else {
        mpInit();
    }
    //关闭图片
    videoParams.closeImgBase64 = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABgAAAAYCAYAAADgdz34AAAABGdBTUEAALGPC/xhBQAAACBjSFJNAAB6JQAAgIMAAPn/AACA6QAAdTAAAOpgAAA6mAAAF2+SX8VGAAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAAsTAAALEwEAmpwYAAAAB3RJTUUH2gQaAAAAtVWE0gAABWZJREFUSMe9VUtvG9cV/s69Qw7HpDiU+JKfQuSYpCgkSM3EtVEEaA3DQX9BN2k2BhIjSBZJvO3Kdjd2ZQeF+0LhhRv0DwRwa7RIFoTgGvFDBlxblZLYoU3JDh8mZzgznJl753ZBSI6kFMkmPauZO9/5vnvPfOdc4AcO2rzwz1df3QI6Uq9/a/L3wbLNCUpK6LkcFup1AtH/JFpfI8JCvU56Ngsl5RYsbU4QgwFpyeTBeC73itdsXtVzuRtiMIi+ubs1rJZKMb/drhk7dhwKOp3P5HB4jRvGBiyt71wI+O02JYrFn0/Uar8xZ2Yq3Zs3/91dWDgR+f6V2NiYWjsRlEJo28R0/bWJl146m63VZnv37i12r1//wHv8+G+JfF6RpuFIvQ4OAK9PTuLza9eoODv72vgLL5zLFAoVrdtFIp8vSKX2B/3+onCcBzyRAAAE/T7xROKwWS6fL1Yqs3HLgm6aOZVIvCxc9/Pl+fkvxicn8ZdmcyTw1sGDMPP5UrpUujhRKMyI5WUEjx6BXBdj+/blJdHLoWUtV99994v8gQPo3Lhx1CyXPyzOzFTRaCBsNkGOA2PXrlykaftTyeQ/jJ07Oxfv3oUGADHTRGBZOfL97WJlBWo4BHEOadugBw9QKJWqRHTmzvnzAIBMpXImXy5X8egRIscBOIcKQ1CrBRJiOyUSuVg6vQRgJGAtLiKS8o7l+x9RNnt8LJ2OwXEApSAtC9RoIF8qvQiiCwBQKJefp5UVRIMBiAikacC2beg9eRJaq6sfhZ53x3IcrAsQ5yAiSwpxst/tKhAdT5tmHK4LRBGkZUFrNLDn0KHnQQT/9m1EgwHAGIjzEXmrFXRXV/8gguAU0zSLOH8mAACMMQBoRVKe6nc6EYC305lMnDxvJGLbcOfnseYkInpG3m4H3dXV30VC/Jpz3lp32+ZOXvO3lDKrxWK/yuTzb5qZjKEcZwRkDMTYOjmlUnjabntPV1b+FAbBSc55B0QbunlDJ699YIx1ZBie7n399ce+7wNEUEoBUQRE0eiZCJ7nodtsfhwGwWnG2Bbybx0VUAp7jx0DiCr6tm17uaZBhSGUEIikRCTE6D0MoWka9FRqLxGVM9UqlFLfPSqKhw9j5fLlg2alcmG8WNyvvvoKynVHZSFaLxMYA0smoXbuRKfZvNG/d++dTLX6L/vLLzdUg3+TvLuwANHv/8Qsl387MTVVi5aWIDodKCFAjIEZBpimQQUB5HCI0HWBMERy374dirEf9RcX7zy8evVhZmoKb0xN4VKj8cxFpGnIVKs/TpdKc+ndu2ve7duIOh1wxkC6DjYxAS9SEgQYmQxX7TbCMIS7ugoeBBgrl2sgOkdE7xBjn6ko2vgPBvfvJ1PPPfdBateuA/bdu3CfPkUUi4EMA1qxCHs49ForzXPtZvO85boez+ehEgkIzmF3OugvLSG1e/eBsenp4/3lZWONd/0E2vg4yV4vJhqNke8TCShdBx8fh+04rtXtzskwPAOl0G+3hwDeN/J5Y9jrQfk+NADy4UMp+v0ON81oi0AMGAxtew5STqey2Re5riMej2MwGHh2t3s2knKOc24BQCTE2X67HSngRNI0jbgQ0JSC9fjx373B4PfJbNZf4+UA8MbUFCgeh5FMNoaue58z9opuGDnfcQZ2rzcnwvAsY8wCEUAEAoYiDK/LICCN81pc0+J2qzXvOc77IPoPjaYCLjUaW22qlCKuaUeJ6JeRlHURhn9ljNnE2IYbTUURpJRjcV3/BTE2K8LwUkzXb4kg2Ni0m7uYiFS2ULjys08/fX37nj1/XCPfHMQYGOf25PT0n3/6ySfvxeLxW6Hvb+D6v8R/AXsDpx5ADBWAAAAAJXRFWHRkYXRlOmNyZWF0ZQAyMDE4LTA2LTI4VDIxOjMxOjMyKzA4OjAwb7VTZgAAACV0RVh0ZGF0ZTptb2RpZnkAMjAxMC0wNC0yNlQwMDowMDowMCswODowMKBVO10AAABDdEVYdHNvZnR3YXJlAC91c3IvbG9jYWwvaW1hZ2VtYWdpY2svc2hhcmUvZG9jL0ltYWdlTWFnaWNrLTcvL2luZGV4Lmh0bWy9tXkKAAAAGHRFWHRUaHVtYjo6RG9jdW1lbnQ6OlBhZ2VzADGn/7svAAAAF3RFWHRUaHVtYjo6SW1hZ2U6OkhlaWdodAAzMij0+PQAAAAWdEVYdFRodW1iOjpJbWFnZTo6V2lkdGgAMzLQWzh5AAAAGXRFWHRUaHVtYjo6TWltZXR5cGUAaW1hZ2UvcG5nP7JWTgAAABd0RVh0VGh1bWI6Ok1UaW1lADEyNzIyMTEyMDAJKQ3vAAAAEXRFWHRUaHVtYjo6U2l6ZQA0MDc1Qi4coXAAAABedEVYdFRodW1iOjpVUkkAZmlsZTovLy9ob21lL3d3d3Jvb3QvbmV3c2l0ZS93d3cuZWFzeWljb24ubmV0L2Nkbi1pbWcuZWFzeWljb24uY24vc3JjLzEwOS8xMDkxNS5wbmdGrRjUAAAAAElFTkSuQmCC";
    //放大
    videoParams.maxImgBase64 = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABgAAAAYCAQAAABKfvVzAAAABGdBTUEAALGPC/xhBQAAACBjSFJNAAB6JgAAgIQAAPoAAACA6AAAdTAAAOpgAAA6mAAAF3CculE8AAAAAmJLR0QA/4ePzL8AAAAHdElNRQffCBAWNRGf/yWCAAAAz0lEQVQ4y7WTuw2DMBRFjwiRqCxReZVIFOzBCAxB1gCPkCaLsEEqulSUSAgLpwgfp7Ffk+vK0jl60rs2xFIw4Laz8IjyVMyH4FguUeHFyo3rdkviE0DTezOiUbRYuaDoWHFYeqa4sOMrLZo7M0MIzzEb3qGAjIoiJJSMHi5ISs2IkeJfpSSX4/9JTkkqxxWGkVqq7DWNlCGsoCLDb9WENzMw06Bpf1oNxOGY6LHSVs+Ha2klrZ5Cj5bs5hQmGrK4sHjKTBXDE57Y4/YOfxCAD41pbIdqglfOAAAAJXRFWHRkYXRlOmNyZWF0ZQAyMDE4LTA2LTI4VDIyOjExOjMxKzA4OjAwu/uc9AAAACV0RVh0ZGF0ZTptb2RpZnkAMjAxNS0wOC0xNlQyMjo1MzoxNyswODowMErQ38kAAABDdEVYdHNvZnR3YXJlAC91c3IvbG9jYWwvaW1hZ2VtYWdpY2svc2hhcmUvZG9jL0ltYWdlTWFnaWNrLTcvL2luZGV4Lmh0bWy9tXkKAAAAGHRFWHRUaHVtYjo6RG9jdW1lbnQ6OlBhZ2VzADGn/7svAAAAGHRFWHRUaHVtYjo6SW1hZ2U6OkhlaWdodAAzNzg9j2GrAAAAF3RFWHRUaHVtYjo6SW1hZ2U6OldpZHRoADM3OK5+MfYAAAAZdEVYdFRodW1iOjpNaW1ldHlwZQBpbWFnZS9wbmc/slZOAAAAF3RFWHRUaHVtYjo6TVRpbWUAMTQzOTczNjc5NwLJOA4AAAARdEVYdFRodW1iOjpTaXplADM4ODlC/mFdvgAAAGJ0RVh0VGh1bWI6OlVSSQBmaWxlOi8vL2hvbWUvd3d3cm9vdC9uZXdzaXRlL3d3dy5lYXN5aWNvbi5uZXQvY2RuLWltZy5lYXN5aWNvbi5jbi9zcmMvMTE5MTYvMTE5MTY5NS5wbmevtH5NAAAAAElFTkSuQmCC";
})();

(function () {
	
	if(jSW){
		if(!jSW._Plugin){
			jSW._Plugin = {};
		}
		if(!jSW._Plugin.ComponentHelper){
			jSW._Plugin.ComponentHelper = function	(args){
				this.cmpName = args.name;
			};

			jSW._Plugin.ComponentHelper.prototype.log = function(text, obj) {

			};
		}
	}
})();
