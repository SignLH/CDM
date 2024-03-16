(function () {
    var multiphelp = null;
    function infolog(des) {
        if (multiphelp === null && jSW._Plugin && jSW._Plugin.ComponentHelper) {
            multiphelp = new jSW._Plugin.ComponentHelper({ name: "GroupCall" });
        }
        if (multiphelp) {
            multiphelp.log(des);
        }
    }

    function checkJswResult(rc) {
        return rc == jSW.RcCode.RC_CODE_S_OK;
    }

    var gutils = {
        guid: function () {
            return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function (c) {
                var r = Math.random() * 16 | 0, v = c == 'x' ? r : (r & 0x3 | 0x8);
                return v.toString(16);
            });
        },
        unfixedConfLS: {
            __index: "jSW_GroupCall_UnfixedConf_Info",
            setInfo: function (confid, users) {
                var confinfo = {
                    confid: confid,
                    users: users
                };
                if (localStorage) {
                    localStorage.setItem(this.__index, this.serializeConfInfo(confinfo));
                }
            },
            getInfo: function () {
                if (localStorage) {
                    return this.parseConfInfo(localStorage.getItem(this.__index));
                }
                return null;
            },
            delInfo: function () {
                if (localStorage) {
                    localStorage.setItem(this.__index, "");
                }
            },
            serializeConfInfo: function (confInfo) {
                var strConfInfo = JSON.stringify(confInfo);
                return strConfInfo;
            },
            parseConfInfo: function (strConfInfo) {
                var confInfo = null;
                try {
                    confInfo = JSON.parse(strConfInfo);
                } catch (e) { }

                return confInfo;
            },
            CheckLocalStorageUnfixedConf: function (afterCheckOver) {
                var confInfo = this.getInfo();
                if (confInfo) {
                    //confInfo.confid
                    var lastConf = gparams.confmanager.swGetConfByConfId(confInfo.confid);
                    if (lastConf) {
                        deleteConf(confInfo.confid, function () { });
                    }
                    if (typeof confInfo == Array) {
                        beforeCloseConfPushBackUsers(confInfo.users);
                    }
                    this.delInfo();
                }
            }
        }
    }

    var gparams = {
        defaultconfname: "UnFixed-",
        getNotFixedConfName: function () {
            return this.defaultconfname + gutils.guid();
        },
        desdiv: null,
        desSession: null,
        UserInfosOrConfId: null,
        szOfflieUser: null,
        szUserBeAboutToInvite: null,
        szUserFiltedBeAboutToInvite: null,
        callback: null,
        tag: null,
        confmanager: null,
        desconf: null,
        getDefaultConf: function () {
            if (this.desconf) {
                var conf = this.confmanager.swGetConfByConfId(
                  this.desconf.swGetConfInfo().id
                );
                return conf;
            }
        },
        isinited: false,
        bbusy: false,
        checkUserContain: function (puid) {
            for (iiindex in this.UserInfosOrConfId) {
                if (this.UserInfosOrConfId[iiindex].puid == puid) {
                    return true;
                }
            }
            return false;
        },
        fillInviteUser: function (szUser) {
            this.inviteResult = [];
            this.szOfflieUser = [];
            this.szUserFiltedBeAboutToInvite = [];
            this.szUserBeAboutToInvite = szUser;
            for (var iindex in szUser) {
                if (this.checkUserContain(szUser[iindex].id)) {
                    this.szUserFiltedBeAboutToInvite.push(szUser[iindex]);
                }
            }
        },
        checkOnUserOnline: function (bOnline, userId) { },
        getNextInviteUser: function (user) {
            if (this.szUserFiltedBeAboutToInvite.length == 0) {
                return null;
            }
            if (user == null) {
                return this.szUserFiltedBeAboutToInvite[0];
            }
            var iindex = this.szUserFiltedBeAboutToInvite.indexOf(user);
            iindex++;
            if (iindex >= 0 && iindex < this.szUserFiltedBeAboutToInvite.length) {
                return this.szUserFiltedBeAboutToInvite[iindex];
            }
            return null;
        },
        checkBNeedToInvite: function (puid, callback, tag) {
            if (!this.checkUserContain(puid) || !this.desconf) {
                callback(false, tag);
            } else {
                var mytag = {
                    callback: callback,
                    tag: tag,
                    puid: puid
                };
                this.getDefaultConf().swGetParticularList({
                    callback: function (options, response, szUser) {
                        var tag = options.tag;
                        var callback = tag.callback;
                        var puid = tag.puid;
                        tag = tag.tag;
                        var bContain = false;
                        for (var iIndex in szUser) {
                            if (szUser[iIndex].id == puid) {
                                bContain = true;
                                break;
                            }
                        }

                        callback(response.emms.code == jSW.RcCode.RC_CODE_S_OK && !bContain, tag);
                    },
                    tag: mytag
                });
            }
        },
        inviteResult: [],
        onGetInviteResult: function (user, bresult) {
            var inviteresult = {
                puid: user.id,
                bresult: bresult
            };
            this.inviteResult.push(inviteresult);
        },
        getInviteResult: function () {
            return this.inviteResult;
        },
        clear: function (onClearHasResult) {
            if (gparams.bbusy) {
                onClearHasResult(jSW.RcCode.RC_CODE_E_BUSY);
            } else {
                if (gparams.checkIsFixedGroup()) {
                    var rc = jSW.RcCode.RC_CODE_E_FAIL;
                    if (this.desconf) {
                        var currentUser = this.desconf.swGetCurrentUser();
                        rc = this.desconf.swParticipatorLeave({
                            user: currentUser,
                            callback: function (options, response) {
                                options.tag(jSW.RcCode.RC_CODE_S_OK);
                            },
                            tag: onClearHasResult.bind(this)
                        });
                        this.desconf = null; //清除desconf 防止状态改变设备列表被重新刷新
                    }

                    if (!checkJswResult(rc)) {
                        onClearHasResult(jSW.RcCode.RC_CODE_S_OK);
                    }
                } else {
                    var _gcCurrent = this;
                    var confId = this.desconf ? this.desconf.swGetConfInfo().id : null;
                    this.desconf = null;
                    var rc = jSW.RcCode.RC_CODE_E_FAIL;
                    if (confId) {
                        rc = deleteConf(confId, function () {
                            beforeCloseConfPushBackUsers.bind(_gcCurrent)(_gcCurrent.getUsersNeedPullback());
                            onClearHasResult(jSW.RcCode.RC_CODE_S_OK);
                        });
                    }
                    
                    if (!checkJswResult(rc)) {
                        if (confId) {
                            beforeCloseConfPushBackUsers(this.getUsersNeedPullback());
                        }
                        onClearHasResult(jSW.RcCode.RC_CODE_S_OK);
                    }
                }
            }
            return jSW.RcCode.RC_CODE_S_OK;
        },
        checkdesconf: function () {
            if (this.checkIsFixedGroup()) {
                var conf = this.confmanager.swGetConfByConfId(this.UserInfosOrConfId);
                return conf;
            }
            return null;
        },
        checkIsFixedGroup: function () {
            return typeof this.UserInfosOrConfId == "string";
        },
        checkIsNotFixedGroup: function () {
            return !this.checkIsFixedGroup();
        },
        getPuInfoById: function (puid) {
            for (iiindex in this.UserInfosOrConfId) {
                if (this.UserInfosOrConfId[iiindex].puid == puid) {
                    return this.UserInfosOrConfId[iiindex];
                }
            }
            return null;
        },

        _bRegUserOnline: false,
        addUserOnlineListener: function (userOnlineHandler) {
            if (!this.checkIsFixedGroup()) {
                if (!this._bRegUserOnline && this.desSession) {
                    this._bRegUserOnline = true;
                    this.desSession.swAddCallBack('notify', function (sender, cmd, data) {
                        var __userOnlineHandler = userOnlineHandler;
                        if (data.msg == "notify_pu_onoffline") {
                            if (data.content.onlinestatus == 1) {
                                __userOnlineHandler(data.content.content);
                            }
                        }
                    })
                }
            }
        },

        getUsersNeedPullback: function () {
            return this.UserInfosOrConfId;
        },
        afterCreateConfSaveUnfixedConfNessInfo: function (desconf) {
            this.desconf = desconf;
            gutils.unfixedConfLS.setInfo(desconf._conf_base_info.id, this.UserInfosOrConfId);
        },
        beforeCloseConfPushBackUsersDelSavedNessInfo: function () {
            gutils.unfixedConfLS.delInfo();
        },
        confUserListCombine: function (userlist) {
            if (this.checkIsFixedGroup()) {
                return userlist;
            }
            var aimUserlist = [];
            var tempuser = null;
            var tempconfuser;
            for (var iCIndex in userlist) {
                aimUserlist.push(userlist[iCIndex]);
            }


            for (var iIndex in this.UserInfosOrConfId) {
                tempconfuser = null;
                tempuser = this.UserInfosOrConfId[iIndex];
                for (var iCIndex in userlist) {
                    if (userlist[iCIndex].id == tempuser.puid) {
                        tempconfuser = userlist[iCIndex];
                        break;
                    };
                }

                if (tempconfuser == null) {
                    tempconfuser = new jSW.SWCONF_USER();
                    tempconfuser.id = tempuser.puid;
                    tempconfuser.name = tempuser.name;
                }
                aimUserlist.push(tempconfuser);
            }
            return aimUserlist;
        }
    };



    function disableBusy() {
        gparams.bbusy = false;
    }

    function enableBusy() {
        gparams.bbusy = true;
    }

    function mpinit() {
        if (jSW.GroupCall) {
            infolog("Load Init Error, Load Init Cancel");
            return;
        }
        jSW.GroupCall = {
            Init: function (desdiv, session, onGCInitHasResult) {
                if (!desdiv || !session || !onGCInitHasResult) {
                    return jSW.RcCode.RC_CODE_E_INVALIDARG;
                }
                gparams.desdiv = desdiv;
                createGroupCallUI();
                gparams.desSession = session;
                var rc = initJSWConfMgr(onGCInitHasResult);
                return rc;
            },

            _innerInviteProxy: function (UserInfosOrConfId, callback, handler) {
                var rc = jSW.RcCode.RC_CODE_E_FAIL;
                if (!UserInfosOrConfId || !callback || arguments.length != 3) {
                    rc = jSW.RcCode.RC_CODE_E_INVALIDARG;
                    infolog("Init Args Error");
                } else if (!(UserInfosOrConfId instanceof Array) && !(typeof UserInfosOrConfId == "string")) {
                    rc = jSW.RcCode.RC_CODE_E_INVALIDARG;
                    infolog("Second argument is invalid, you should give me a array contains all users you want to invite or a string means conference Id!");
                } else if (gparams.bbusy) {
                    rc = jSW.RcCode.RC_CODE_E_INVALIDARG;
                    infolog("current status is busy, please wait or close the panel");
                } else if (gparams.desdiv == null) {
                    gparams.isinited = false;
                } else {
                    enableBusy();
                    rc = handler.bind(this)();
                    if (rc != jSW.RcCode.RC_CODE_S_OK) {
                        infolog("Call Invite Faild, Not Inited");
                        disableBusy();
                    } else {
                        infolog("Call Invite Success");
                    }
                }
                return rc;
            },
            /**
                                        UserInfosOrConfId: [{}]
                  */
            Invite: function (UserInfosOrConfId, callback, tag) {
                var rc = this._innerInviteProxy(UserInfosOrConfId, callback, function () {
                    gparams.UserInfosOrConfId = UserInfosOrConfId;
                    gparams.callback = callback;
                    gparams.tag = tag;
                    if (gparams.isinited) {
                        uiglobel.clear();
                    }
                    var confmessage =  getconfbyid (UserInfosOrConfId)
                    if(confmessage){
                        uiglobel.lefttopContentnameinput.value = confmessage._conf_base_info.name;
                        uiglobel.lefttopContentIDinput.value = confmessage._conf_base_info.id;
                        uiglobel.lefttopContenttypecminput.checked = confmessage._conf_base_info.bIsDiscussiongroup;
                        uiglobel.lefttopContenttypeprinput.checked = confmessage._conf_base_info.bIsChairman;
                        uiglobel.lefttopContenttypeininput.checked = confmessage._conf_base_info.bIsInvite;
                        uiglobel.lefttopContenttypenoinput.checked = confmessage._conf_base_info.bIsPassword;
                    }
                    return buildGroupCall();
                });
                return rc;
            },

            Reset: function (onResetHasResult) {
                var rc = gparams.clear(function (rc) {
                    uiglobel.clear();
                    if (onResetHasResult) {
                        onResetHasResult(rc);
                    }
                });
                return rc;
            },
            Clear: function () {
                var rc = this.Reset(function () {
                    uiglobel.remove();
                    gparams.desdiv = null;
                });
                return rc;
            },openCall:function(){
				applySpeak();
			},EndCall:function(){
			    applyStopSpeak();
			}
        };

        infolog("Load Init OK");
    }
    if (typeof jSW == "undefined") {
        infolog(
          "Load Faild! jSW is undefined, Please load jSW.js before load this"
        );
    } else {
        mpinit();
    }
    function getconfbyid (confid){
        confmanager = gparams.desSession.swGetConfManager();
        var conf =  confmanager.swGetConfByConfId(confid);
        if (conf == null) {
           
            return null;
        }
        return conf
    }

    function setmeetingbtncb(bresult){
        uiglobel.leftbottombotton.innerHTML =  bresult.tag._conf_base_info.bIsStarted? "结束会议":"开始会议";
    }

    // var currentGroupCall = this;
    // var confnowID = gparams.UserInfosOrConfId;
    function meetingstart(){
        confmanager = gparams.desSession.swGetConfManager();
        var conf =  confmanager.swGetConfByConfId(gparams.UserInfosOrConfId);
        if (conf == null) {
            console.warn("会议不存在");
            return;
        }
        if(conf._conf_base_info.bIsStarted){
            var rc = conf.swConfStop({
                callback:setmeetingbtncb,
                tag:conf
            });
            return rc;
        }else{
            var rc = conf.swConfStart({
                callback: setmeetingbtncb,
                tag:conf
            });
            return rc;
        }
    }

    function initJSWConfMgr(onGCInitHasResult) {
        gparams.confmanager = gparams.desSession.swGetConfManager();
        var rc = jSW.RcCode.RC_CODE_E_FAIL;
        if (gparams.confmanager) {
            var params = {
                callback: function (rc) {
                    onInitConfHasResult(rc);
                    onGCInitHasResult(rc);
                }
            };
            rc = gparams.confmanager.swInit(params);
        }
        return rc;
    }

    function buildGroupCall() {
        var rc = checkDisplayOrCreate();
        return rc;
    }

    function onInitConfHasResult(rc) {
        if (checkJswResult(rc)) {
            var rc = gparams.confmanager.swRegConfWatch(onNotifyConfStatusChange);
            if (checkJswResult(rc)) {
                gutils.unfixedConfLS.CheckLocalStorageUnfixedConf();
            } else {
                onAsyncInviteHasResult(false, "reg conf change faild:" + rc);
            }
        } else {
            onAsyncInviteHasResult(false, "init conf manager faild:" + rc);
        }
    }

    function onInviteGetResult(options, response, inviteResult) {
        var rc = response.emms.code;
        if (!checkJswResult(rc)) {
            //onAsyncInviteHasResult(false, "no invite user callback faild" + response.emms.code);
        }

        {
            var desconf = options.tag.desConf;
            var desuser = options.tag.desusers;
            gparams.onGetInviteResult(desuser, checkJswResult(rc));
            if (!inviteusers(desconf, desuser)) {
                onAsyncInviteHasResult(false, "no invite user" + rc);
            }
        }
    }

    function inviteusers(confCreated, user) {
        gparams.desconf = confCreated;
        var desusers = gparams.getNextInviteUser(user);
        if (user == null && desusers == null) {
            onAsyncInviteHasResult(true, "no users can invite" + rc);
            return true;
        }
        if (user && desusers == null) {
            onAsyncInviteHasResult(true, "users inviting is over" + rc);
            return true;
        }

        var rc = inviteUser(confCreated, onInviteGetResult, desusers);
        return rc;
    }

    function inviteUser(conf, callback, user) {
        var params = {
            users: [user],
            callback: callback,
            tag: {
                desConf: conf,
                desusers: user
            }
        };
        var rc = conf.swParticipatorAdd(params);
        return checkJswResult(rc);
    }

    function pushUserBackHisConfBelong(szUsers) {
        if (gparams.confmanager) {
            var user = null;
            for (var iIndex in szUsers) {
                user = szUsers[iIndex];
                pushAUserBackHisConf(user);
            }
        }
    }

    function pushAUserBackHisConf(user) {
        var confid = user.confid;
        var desconf = gparams.confmanager.swGetConfByConfId(confid);
        desconf.swGetParticularList({
            callback: function (options, response, szConfUsers) {
                for (var iCuIndex in szConfUsers) {
                    if (szConfUsers[iCuIndex].id == options.tag.id) {
                        if (szConfUsers[iCuIndex].isonline) {
                            inviteUserSpeak(options.tag.desconf, szConfUsers[iCuIndex]);
                        }
                        break;
                    }
                }
            },
            tag: {
                id: user.puid,
                desconf: desconf
            }
        });
    }

    function inviteUserSpeak(desconf, user) {
        desconf.swInviteSpeak({
            user: user,
            callback: function (options, response) {
                infolog(options.tag + response.emms.code);
            },
            tag: user.id + " be push back to " + "conf: " + desconf.swGetConfInfo().id + " "
        })
    }

    function beforeCloseConfPushBackUsers(users) {
        if (!gparams.checkIsFixedGroup()) {
            // pushUserBackHisConfBelong(users);
            // gparams.beforeCloseConfPushBackUsersDelSavedNessInfo();
            infolog("--------------------- UnFixed Conf Close ----------------");
        } else {
            infolog("--------------------- Fixed Conf Close ----------------");
        }
    }

    function unFixedConfAddUserHasResult(options, response, data) {
        infolog("onUserOnlineHandler invite user " + options.tag.desusers.id + checkJswResult(response.emms.code));
    }

    function onUserOnlineHandler(swpu) {
        gparams.checkBNeedToInvite(swpu._info_pu.id, function (bNeedInvite, swpu) {
            if (bNeedInvite) {
                var puUser = {
                    id: swpu._info_pu.id,
                    pid: 0
                };
                var rc = inviteUser(gparams.getDefaultConf(), unFixedConfAddUserHasResult, puUser);
                infolog("onUserOnlineHandler invite user " + swpu._info_pu.id + checkJswResult(rc));
            }
        }.bind(this), swpu);
    }

    function onGetOnlineUser(options, response, users) {
        if (checkJswResult(response.emms.code)) {
            gparams.fillInviteUser(users);
            var confdes = options.tag;
            if (!inviteusers(confdes, null)) {
                onAsyncInviteHasResult(
                  false,
                  "get online users from server success, but invites faild!" + rc
                );
            }
        } else {
            onAsyncInviteHasResult(
              false,
              "conf create success, but it reply faild when get online users from server:" +
                rc
            );
        }
    }

    function getOnlineUser(confCreated) {
        var params = {
            callback: onGetOnlineUser,
            tag: confCreated
        };
        var rc = confCreated.swGetOnlineUsers(params);
        return checkJswResult(rc);
    }

    function onCreateConfHasResult(options, response, desConf) {
        var rc = response.emms.code;
        if (checkJswResult(rc)) {
            var confname = options.tag;
            var confCreated = desConf;
            if (!confCreated) {
                onAsyncInviteHasResult(false, "conf create success, but can not find it by confmanager");
                return;
            }
            gparams.afterCreateConfSaveUnfixedConfNessInfo(confCreated);
            gparams.addUserOnlineListener(onUserOnlineHandler);
            if (!getOnlineUser(confCreated)) {
                onAsyncInviteHasResult(false, "conf create success, but it is faild when call get online users immediately:" + rc);
            }
        } else {
            onAsyncInviteHasResult(false, "conf create faild");
        }
    }

    function checkDisplayOrCreate() {
        var rc = jSW.RcCode.RC_CODE_E_FAIL;
        if (gparams.checkIsFixedGroup()) {
            var conftemp = gparams.checkdesconf();
            if (conftemp) {
                infolog("---------------------- Fixed Display ------------------");
                gparams.desconf = conftemp;
                rc = makeSureIWasInSeat();
            } else {
                infolog("---------------------- Fixed Conf NotFound ------------------");
                rc = jSW.RcCode.RC_CODE_E_NOTFOUND;
            }
        } else if (gparams.checkIsNotFixedGroup()) {
            infolog("---------------------- Unfixed Conf Create ------------------");
            rc = createUnfixedConf();
        }
        return rc;
    }

    function makeSureIWasInSeat() {
        var rc = jSW.RcCode.RC_CODE_S_OK;
        var currentUser = gparams.desconf.swGetCurrentUser();
        if (currentUser.isinseat) {
            onAsyncInviteHasResult(true, "Fiexd Group Display");
        } else {
            rc = returnToDefConf();
        }
        return rc;
    }

    function onReturnToDefConfHasResult(options, response, data) {
        onAsyncInviteHasResult(
          checkJswResult(response.emms.code),
          "Fiexd Group Display"
        );
    }

    function returnToDefConf() {
        var returnParam = {
            user: gparams.desconf.swGetCurrentUser(),
            callback: onReturnToDefConfHasResult,
            tag: null
        };
        var rc = gparams.desconf.swParticipatorReturn(returnParam);
        return rc;
    }

    function deleteConf(id, onDelteHasResult) {
        var params = {
            confid: id,
            callback: onDelteHasResult,
            tag: null
        };
        var rc = gparams.confmanager.swDeleteConf(params);
        return rc;
    }

    function createUnfixedConf() {
        var confbaseinfo = {
            name: gparams.getNotFixedConfName(),
            speakmode: jSW.SwConfManager.MODE_SPEAK.CHAIRMAN,
            joinmode: jSW.SwConfManager.MODE_JOIN.INVITE,
            applyformode: jSW.SwConfManager.MODE_APPLY.AUTOAGREE,
            startmode: jSW.SwConfManager.MODE_START.FOREVER,
            itimeout: 24 * 60 * 60
        };
        var params = {
            confbaseinfo: confbaseinfo,
            callback: onCreateConfHasResult,
            tag: confbaseinfo.name
        };
        var rc = gparams.confmanager.swCreateConf(params);
        return rc;
    }

    function onAsyncInviteHasResult(bresult, des) {
        if (gparams.callback) {
            var rc = bresult ? jSW.RcCode.RC_CODE_S_OK : jSW.RcCode.RC_CODE_E_FAIL;
            gparams.callback(rc, gparams.tag, gparams.getInviteResult());
        }
        if (!bresult) {
            gparams.desconf = null;
            infolog(des);
            disableBusy();
        } else {
            getDefaultUserlistAndShow();
            uiglobel.enable();
            disableBusy();
        }
    }

    function getDefaultUserlistAndShow() {
        if (gparams.desconf) {
            var params = {
                callback: onGetDefaultUserlist,
                tag: this
            };
            gparams.desconf.swGetParticularList(params);
        }
    }

    function onGetDefaultUserlist(options, response, userlist) {
        if (checkJswResult(response.emms.code)) {
            var desuserlist = gparams.confUserListCombine(userlist);
            showUser(desuserlist);
        } else {
            infolog("get userlist error");
        }
    }

    var uiglobel = {
        containerPanel: document.createElement("div"),
        containerPanelReset: document.createElement("img"),
        containerPanelResetImgSrc:
          "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADAAAAAwCAMAAABg3Am1AAAABGdBTUEAALGPC/xhBQAAACBjSFJNAAB6JgAAgIQAAPoAAACA6AAAdTAAAOpgAAA6mAAAF3CculE8AAABYlBMVEUAAADTLy/TLy/TLy/TLy/TLy/0Qzb0Qzb0Qzb0Qzb0QzbTLy/TLy/TLy/TLy/TLy/TLy/TLy/0Qzb0Qzb0Qzb0Qzb0Qzb0Qzb0QzbTLy/TLy/TLy/TLy/TLy/TLy/TLy/TLy/0Qzb0Qzb0Qzb0Qzb0Qzb0Qzb0Qzb0QzbTLy/TLy/TLy/TLy/TLy/TLy/0Qzb0Qzb0Qzb0Qzb0Qzb0QzbTLy/TLy/TLy/TLy/TLy/TLy/TLy/0Qzb0Qzb0Qzb0Qzb0Qzb0Qzb0QzbTLy/TLy/TLy/TLy/0Qzb0Qzb0Qzb0QzbTLy/TLy/TLy/0Qzb0Qzb0QzbSLi/1RDbTLy/MKy77Rzf0QzbTLy/UMC/zQjb0QzbTLy/XMTDwQTX0QzbTLy/TLy/TLy/TLy/XMTDwQTX0Qzb0Qzb0Qzb0QzbTLy/TLy/TLy/TLy/TLy/0Qzb0Qzb0Qzb0Qzb0QzbXMTDwQTX///+DMjEuAAAAc3RSTlMAAAAAAAAAAAAAAAAAAAsGAAAAAAYLAAAAAAAao3gGAAAAAAZ4oxoAAACr//Z3BQV39v+rAAD///V3BQAABXf1//8A///1d3f1//8A/////wAAAHcFBXf1e3v1//r6/3f1////////9XcABXv6///6ewUAJoszcwAAAAFiS0dEdahqmPsAAAAHdElNRQfeDAoPOSh+wLjnAAABVUlEQVRIx2NgGAWDDDAyMbOwYgqzsXNwcmFTz83Dy8cvIIguLCQsIiomLoGpXpJHSlpGVk5eAVVYUUlZRVVNXVwDQ4Mmr7SWto6unjyKHUJK+gaGRsZqYiYYGkz5ZLTNzC0srayRdAjZ2NrZOzgaqYo6YWhg4ZfVMXd2cXWzQrhKUcnW3cPTy8FQRYQdQwOrt5yuhYuPr6sl3FVA99h5+Pl72hsoB7Bh+lpQXs/S1dfHBe4qsHs8/f087PSVhLCFq4K8lZurC9xVUPd4erjbKilijzlBaytLuKvg7rGztRFiwAGQXBUYRMA9aK4KDgkNI+AeVFeFR0RGEXIPiquiY2LjCLoH7qr4hMSk5JTUtPQMAu6B2pGZlZ2TW1ySl19QWETQfDI0kOokUj1NarCSGnGkJg1SEx+pyZvUDERyFiW5ECC5mCG5ICO5qCS5MCa9uB8FAwkA1tqHz8dmcxMAAAAldEVYdGRhdGU6Y3JlYXRlADIwMTgtMDYtMjhUMjI6MDY6NDYrMDg6MDBXy2/aAAAAJXRFWHRkYXRlOm1vZGlmeQAyMDE0LTEyLTEwVDE1OjU3OjQwKzA4OjAwk6VBzAAAAEN0RVh0c29mdHdhcmUAL3Vzci9sb2NhbC9pbWFnZW1hZ2ljay9zaGFyZS9kb2MvSW1hZ2VNYWdpY2stNy8vaW5kZXguaHRtbL21eQoAAAAYdEVYdFRodW1iOjpEb2N1bWVudDo6UGFnZXMAMaf/uy8AAAAYdEVYdFRodW1iOjpJbWFnZTo6SGVpZ2h0ADEyOEN8QYAAAAAXdEVYdFRodW1iOjpJbWFnZTo6V2lkdGgAMTI40I0R3QAAABl0RVh0VGh1bWI6Ok1pbWV0eXBlAGltYWdlL3BuZz+yVk4AAAAXdEVYdFRodW1iOjpNVGltZQAxNDE4MTk4MjYwhZrzzwAAABB0RVh0VGh1bWI6OlNpemUAODU5Qrs5NVsAAABidEVYdFRodW1iOjpVUkkAZmlsZTovLy9ob21lL3d3d3Jvb3QvbmV3c2l0ZS93d3cuZWFzeWljb24ubmV0L2Nkbi1pbWcuZWFzeWljb24uY24vc3JjLzExODE3LzExODE3NTUucG5nDRejrAAAAABJRU5ErkJggg==",

        minPanelReset: document.createElement("img"),
        minPanelResetImgSrc:
            "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADAAAAAwCAYAAABXAvmHAAAA1ElEQVRoQ+2XQQ3DQAwEbQaBUgiFYApB0kIpBEMJhFAIAkd99B1LE+l06uZ7t5t4xp+4Tf745N9vGmC0QRmQAUhAKwQB4rgMYISwQAYgQByXAYwQFsgABIjjMoARwgIZgABx/H8MRMTT3QsjaxRU1ZGZW+Nq76c+It7u/uoU3nWnqtbM/Fz1tVZo+gG+FCLiYWbLFZGbzvfM3DtdLQOdolF3NMAo8r/3yoAMQAJaIQgQx2UAI4QFMgAB4rgMYISwQAYgQByXAYwQFsgABIjjMoARwoITdWUeMQfT9akAAAAASUVORK5CYII=",

        containerPanelMask: document.createElement("div"),
        leftPanel: document.createElement("div"),
        leftPanelContentItem: document.createElement("div"),

        lefttopContent:document.createElement("div"),
        
        lefttopContentname:document.createElement("div"),
        lefttopContentnamespan:document.createElement("span"),
        lefttopContentnameinput:document.createElement("input"),
        lefttopContentIDspan:document.createElement("span"),
        lefttopContentIDinput:document.createElement("input"),

        lefttopContenttype:document.createElement("div"),
        lefttopContenttypecmspan:document.createElement("span"),
        lefttopContenttypecminput:document.createElement("input"),
        lefttopContenttypeprspan:document.createElement("span"),
        lefttopContenttypeprinput:document.createElement("input"),
        lefttopContenttypeinspan:document.createElement("span"),
        lefttopContenttypeininput:document.createElement("input"),
        lefttopContenttypenospan:document.createElement("span"),
        lefttopContenttypenoinput:document.createElement("input"),

        leftPanelContent: document.createElement("div"),
        leftPanelContentPic: document.createElement("img"),

        leftPanelContentItemHandle: document.createElement("div"),
        leftPanelContentItemContainer: document.createElement("div"),
        leftPanelContentBtnApply: document.createElement("button"),
        leftPanelContentBtnStatus: document.createElement("img"),

        leftbottombotton:document.createElement("button"),

        leftPanelContentPicImgBase64:
          "data:image/png;base64,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",

        rightPanel: document.createElement("div"),
        style: document.createElement("style"),
        applyimgsrc:
          "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABgAAAAYCAYAAADgdz34AAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAAJcEhZcwAADsMAAA7DAcdvqGQAAARlSURBVEhLnVZbU1tVFE596vjk+AN88NkHf4QvVivaC7WlDONIW5kCLZKc5CSkoaAPdtRx2qovigW1apVL7lAoCVBghArknJNwCQQICCnUB7BNaUE+19o5NCEJw+iaWXNyZq/1fXutvb91YtjPSr4LHJT9k+dk33iPyaltyr4JWNzhh/TutnfOHNXD/rtZflNetrZHv5D94wsMSoAwu7VnLog84W1rx9S41T9hr/pl4EU9dX8zeyJV1o7ptbq+Zdg6ojB7wilgelqYSH8nAtR0zqDubgK29ql5ya0d0yH2tANmb/iGngCzS00B01P2T4jf1b8Ow+KNpIjc6XV7VwyOQBySU7PrWLkmOZXG+v4VsTORqLuF2lF1sx+F5fUoKK7GKeNnMLWMCKJncbwJiuOqjW2KRYdMm6ktdIF3ng3OToeMk8ZPcejoB3irqAqHjpzFe/UNsN6ezo0lkkt35hjnNR2aeu7SXpJ9kSQdak6C6LsnguPn63H4ZKWo4M0T5ThtuQorn092PFXCBCa3EnUEZg8KAsmlXq3rS6R7nuk6QWHFx7sIiq3X8xPozq2SnOr7hrOue8+b3FpC3JY8gf+X4FL3PD+D1B61lPtm9uQGmagio4uv494E1U4VEsVk5wpMt7JKBKH+bBGxc5LdF4Gjg9fCeQlkOrPLtG71hkEayMGgc3hiYPlnL7CXN4/h9uR9hBPrqGxVcwiKiMDsn8LS2mM0DcdxoSWUg0Hd2TakSslaIGeCvtgDxP5KoqJVIQ18tJtAvgYLiXFtYxM/jSyiMh8BOd+gZL4FJuiiCpbXN3DRqYkKGJgJ3jhWRhV8SWNiGsmnW2gcms9P4FS2DSR3384YyPSLbSq+Ck6A7UrPLIo+uUkiu4jD71agoMSE4mseNI0uA/9soc6n0mHvzheCdSobBtkVflvMliwFS/RecWsYU3+uYDW5BZtXw6nPm1FY14DT1z24EohicxsIaHM4f+sPytmdbyX1Sy5lyeBw4DlqU4ynYmYA+4d0BU0/30VsaQVPCWwgvoZmdQmjiYeisqGJOZT/2E+3JXe8OIILfM09KSXTBOQ5lK1krqqKDriiMYjvu0cwt7wqgLXYIr72/46yph6hk8wc4aSp2uAitShcIAgcraMvUEsSNZ0xCsgl4R2e+KYX9pYh/P34Cc7c6EFJ4yABZUxT3akbAlxqGxswAAcEAZuxbfQdR2BBV2A6gdXMQvJPPkDo/iNRweDCOlrUhFjbJTB6r6EJS0Nzo9o99ooOnTYa2TU8pMSt0ttlohbYqOShyThC03EMqlGoMwu4o87CmDEmeOc2Ard3zcLkVI7rkLkmuRXZTuOWR+7Ozjj5XGMvzjR0o/TbLvEs+6Gf1tKHW9uzSN+H6CP6aBXqUHub5Aq9Tt9k7XLvkpiK3DaLdzzlNLd2fvNV5NtSS25rjwaNraOv6hD7G/9doR2VWtunusze8QS1bINni9gxKdTi0ZJEEqd/Fa2yJ3JET8syg+Ff85596LwlazMAAAAASUVORK5CYII=",
        applyendimgsrc:
          "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABgAAAAYCAYAAADgdz34AAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAAJcEhZcwAADsMAAA7DAcdvqGQAAAN+SURBVEhLnVZLS1tREI5dSVelP6CLrrvoj+imqzZJGxHdVYgrQ9VFilXBhRsXxvhCRE18v6Mmxk0XCdiCgvggEjRSSKJFKSjFNhjUjN93uImXm5tKOzCce86Z+b6ZOXNOYnlIWltby6enp52Tk5PR0dHR66mpKZmYmPiNeXB+ft6umf27DA0NPQewB5omKHV8fLyg2jyH/Ti+mwcHB59qrg8LHD/Ozs7+CgaDgpERK1COWgaFObKQUCgkIEqOjY290yBKShmcfHSYmZkpREsFgBqRmaA8RRktLCxIIBAQkDRrWMWCTf/q6mohwrwSbGBgQGpra6WqqkpcLpf4fD5FZLRj1n6/361B3gvAXYzcCE5l9HV1dWKz2aSyslKsVqs0NTWp8hltSbK4uEicVxq0xQLGZ4gmYywLlYRUp9MpFRUVKgOHwyH19fWmBFQSYEyMjIyUKwK0n5epGQ2peQKWR0/Q2NhYkoBKPOB+sKC2j1Ge01LG/0vALIAbYe1r9K2nV+wp/RsBolQ2Rl9iYv0nCb5yYjSgEw93bm5Ozc0IeGbcN/OnAiPL+l+bbbLfd3d3JZ1OCw6riKChoUEBn5+fSzQaVTZGDBDkLKXYSRCPx+Xs7KwkATPMZDKyvr4uw8PDpjjMIGO2QYK9vT25uLjgxVEEBCaB3W5XBCxPNpuVSCRiSgDsHM8gnH8G9MqbGg6HhcKWa2lpUZeMWVRXV0t7e7uK/Pb2Vp0FD1vvz8YA9hUzeMsrb+wizvFCSiqVksvLS1XvtrY2cbvdCnxlZUWBb29vKzu9L5VBA/sH3/tHYPrOV9FoxKjofHx8LDc3N5JIJGRjY0OSyaTKLBaLSX9/f5EfdWlpif4hdZNB0Mx3yGjELFiq3t5eWVtbk5OTEwVMIr6cPT09qp3N/EiAs3ujCNAlT7BxapZFvnQdHR0qI3aN1+uVrq6uorJSSahF/w2xlCkCCiK1Mipj29KBa/k7QTk6OpLNzU3T6NlZqP8VCF5o0PeCxc/sGH1XEYRNsL+/LwcHB7KzsyOHh4eytbWlMsqTcCQ4f3iw/l6DLBYYf6KR9uQWnLu7u8Xj8UhnZ6ca+/r6CvvU5eVltusf2Do0qNICh9cwjrEVScQSMQuq/puZst5UfEdwli81iIeFf1fwWtaA6AsO8xSRXUFzjBYl4JjBegr7AXzbNDeDWCx39PBeqINA/jcAAAAASUVORK5CYII=",
        apply: function () {
            uiglobel.leftPanelContentBtnStatus.src = this.applyimgsrc;
            this.leftPanelContentBtnApply.innerHTML = "放开结束发言";
        },
        disapply: function () {
            uiglobel.leftPanelContentBtnStatus.src = this.applyendimgsrc;
            this.leftPanelContentBtnApply.innerHTML = "申请发言";
        },
        init: function () {
            this.disapply();
        },

        userHead: document.createElement("div"),
        userContent: document.createElement("div"),
        userContentUl: document.createElement("ul"),
        userlist: [],
        clear: function () {
            this.userlist = [];
            this.userContentUl.innerHTML = "";
            this.disable();
        },
        enable: function () {
            this.containerPanelMask.style.display = "none";
        },
        disable: function () {
            this.containerPanelMask.style.display = "block";
        },
        remove: function () {
            if (gparams.desdiv) {
                gparams.desdiv.removeChild(this.containerPanel);
            }
        }
    };

    function onClickReset() {
         jSW.GroupCall.Reset(function () { 
		  $("#hndCallDiv").hide();
		});
    }
    
    function onClickMin() {
	   $("#hndCallDiv").hide();
	   $("#callMax").show();
    }
    
    function createGroupCallUI() {
        if (gparams.desdiv) {
            gparams.desdiv.appendChild(uiglobel.containerPanel);
            uiglobel.containerPanel.style.boxSizing = "border-box";
            uiglobel.containerPanel.style.position = "relative";
            uiglobel.containerPanel.appendChild(uiglobel.leftPanel);
            uiglobel.containerPanel.appendChild(uiglobel.rightPanel);
            uiglobel.containerPanel.className = "group-call-plugin";
            uiglobel.leftPanel.className = "group-call-plugin-left";
            uiglobel.rightPanel.className = "group-call-plugin-right";
            uiglobel.containerPanel.style.height = "100%";
            uiglobel.containerPanel.style.width = "100%";
            uiglobel.leftPanel.style.height = "100%";
            uiglobel.leftPanel.style.width = "calc(100% - 300px)";
            uiglobel.rightPanel.style.height = "100%";
            uiglobel.rightPanel.style.width = "300px";
            uiglobel.leftPanel.style.float = "left";
            uiglobel.rightPanel.style.float = "left";
			
			gparams.desdiv.appendChild(uiglobel.containerPanel);
            uiglobel.containerPanel.appendChild(uiglobel.containerPanelReset);
            uiglobel.containerPanelReset.src = uiglobel.containerPanelResetImgSrc;
            uiglobel.containerPanelReset.onclick = onClickReset;
            uiglobel.containerPanelReset.style.position = "absolute";
            uiglobel.containerPanelReset.style.top = "0px";
            uiglobel.containerPanelReset.style.right = "0px";
            uiglobel.containerPanelReset.style.cursor = "pointer";
            uiglobel.containerPanelReset.title = "关闭";
            uiglobel.containerPanelReset.onmouseover = function () {
                this.style.backgroundColor = "white";
            }
            uiglobel.containerPanelReset.onmouseout = function () {
                this.style.backgroundColor = "";
            }
            
            uiglobel.containerPanel.appendChild(uiglobel.minPanelReset);
            uiglobel.minPanelReset.src = uiglobel.minPanelResetImgSrc;
            uiglobel.minPanelReset.onclick = onClickMin;
            uiglobel.minPanelReset.style.position = "absolute";
            uiglobel.minPanelReset.style.top = "0px";
            uiglobel.minPanelReset.style.right = "55px";
            uiglobel.minPanelReset.style.cursor = "pointer";
            uiglobel.minPanelReset.title = "最小化";
            uiglobel.minPanelReset.onmouseover = function () {
                this.style.backgroundColor = "white";
            }
            uiglobel.minPanelReset.onmouseout = function () {
                this.style.backgroundColor = "";
            }
            
            
            uiglobel.leftPanel.appendChild(uiglobel.leftPanelContent);
            uiglobel.leftPanel.style.textAlign = "center";
            uiglobel.leftPanel.style.backgroundColor = "aliceBlue";
            uiglobel.leftPanel.style.position = "relative";

            uiglobel.leftPanel.appendChild(uiglobel.lefttopContent);
            uiglobel.lefttopContent.style.textAlign = "left";
            uiglobel.lefttopContent.style.lineHeight = "20px;";
            uiglobel.lefttopContent.className = "confmessagetop";

            
            uiglobel.leftPanelContent.appendChild(uiglobel.leftPanelContentItem);
            uiglobel.leftPanelContent.style.position = "relative";
            uiglobel.leftPanelContent.style.top = "40%";

            uiglobel.leftPanelContentItem.appendChild(
              uiglobel.leftPanelContentItemContainer
            );

            uiglobel.leftPanelContentItemContainer.appendChild(
              uiglobel.leftPanelContentBtnStatus
            );
            uiglobel.leftPanelContentBtnStatus.style.position = "absolute";
            uiglobel.leftPanelContentBtnStatus.style.left = "calc(50% + 40px)";

            uiglobel.leftPanelContentItemContainer.appendChild(
              uiglobel.leftPanelContentPic
            );
            uiglobel.leftPanelContentPic.src = uiglobel.leftPanelContentPicImgBase64;

            uiglobel.leftPanelContentItemContainer.appendChild(
              uiglobel.leftPanelContentItemHandle
            );

            uiglobel.leftPanelContentItemHandle.appendChild(
              uiglobel.leftPanelContentBtnApply
            );
            uiglobel.leftPanelContentItemHandle.style.marginTop = "10px";
            uiglobel.leftPanelContentItemHandle.style.marginBottom = "10px";

            uiglobel.leftPanelContentItemContainer.style.position = "relative";
            uiglobel.leftPanelContentItemContainer.style.marginTop = "-50px";

            uiglobel.init();

            uiglobel.rightPanel.appendChild(uiglobel.userHead);
            uiglobel.rightPanel.style.padding = "20px 10px";
            uiglobel.rightPanel.style.boxSizing = "border-box";
            uiglobel.rightPanel.appendChild(uiglobel.userContent);
            uiglobel.userHead.style.lineHeight = "20px";
            uiglobel.userHead.innerHTML = "用户列表";

            uiglobel.userContent.style.height = "calc(100% - 20px)";
            uiglobel.userContent.style.overflowY = "auto";

            uiglobel.userContent.appendChild(uiglobel.userContentUl);
            uiglobel.rightPanel.style.backgroundColor = "lightGrey";
            uiglobel.userContentUl.style.listStyle = "none";
            uiglobel.userContentUl.style.paddingLeft = "10px";

            uiglobel.leftPanelContentBtnApply.onmousedown = applySpeak;
            uiglobel.leftPanelContentBtnApply.onmouseup = applyStopSpeak;
        }
    }
    

    var UserEle = function (id) {
        this.id = id;
        this.ele = document.createElement("li");
        this.ele1 = document.createElement("img");
        this.elename = document.createElement("span");
        this.ele2 = document.createElement("img");
        this.ele3 = document.createElement("img");
        this.ele.appendChild(this.ele1);
        this.ele.appendChild(this.ele3);
        this.ele.appendChild(this.ele2);
        this.ele.appendChild(this.elename);
        this.ele1.style.margin = "0px 3px";
        this.ele2.style.margin = "0px 3px";
        this.ele3.style.margin = "0px 3px";
        this.ele.style.padding = "5px 0px 3px 0px";
        this.ele.style.margin = "2px 0px";
        this.ele.style.backgroundColor = "white";
        this.ele.style.borderRadius = "5px";
    };

    UserEle.prototype = {
        check: function (user) {
            return user.id == this.id;
        },
        getEle: function () {
            return this.ele;
        },
        fill: function (user) {
            if (user.id.indexOf("CU") > -1) {
                if (user.isonline) {
                    this.ele1.src = UserEle.Images.Mine;
                } else {
                    this.ele1.src = UserEle.Images.CuOffline;
                }
                this.elename.innerHTML = user.name + "(" + user.id + ")";
            } else if (user.id.indexOf("PU") > -1) {
                if (user.isonline) {
                    this.ele1.src = UserEle.Images.PuOnline;
                } else {
                    this.ele1.src = UserEle.Images.PuOffline;
                }
                var puinfo = gparams.getPuInfoById(user.id);
                var puname = user.aliasname;
                if (puinfo) {
                    puname = puinfo.name;
                }
                this.elename.innerHTML = puname + "(" + user.id + ")";
            }

            if (user.isSpeak) {
                this.ele2.src = UserEle.Images.AudioInOn;
                this.ele.style.backgroundColor = "green";
            } else {
                this.ele.style.backgroundColor = "white";
                this.ele2.src = UserEle.Images.AudioInOff;
            }

            if (user.isinseat) {
                this.ele3.src = UserEle.Images.AudioOutOn;
            } else {
                this.ele3.src = UserEle.Images.AudioOutOff;
            }
        }
    };

    UserEle.Images = {
        PuOnline:
          "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAACXBIWXMAAA9hAAAPYQGoP6dpAAAKTWlDQ1BQaG90b3Nob3AgSUNDIHByb2ZpbGUAAHjanVN3WJP3Fj7f92UPVkLY8LGXbIEAIiOsCMgQWaIQkgBhhBASQMWFiApWFBURnEhVxILVCkidiOKgKLhnQYqIWotVXDjuH9yntX167+3t+9f7vOec5/zOec8PgBESJpHmomoAOVKFPDrYH49PSMTJvYACFUjgBCAQ5svCZwXFAADwA3l4fnSwP/wBr28AAgBw1S4kEsfh/4O6UCZXACCRAOAiEucLAZBSAMguVMgUAMgYALBTs2QKAJQAAGx5fEIiAKoNAOz0ST4FANipk9wXANiiHKkIAI0BAJkoRyQCQLsAYFWBUiwCwMIAoKxAIi4EwK4BgFm2MkcCgL0FAHaOWJAPQGAAgJlCLMwAIDgCAEMeE80DIEwDoDDSv+CpX3CFuEgBAMDLlc2XS9IzFLiV0Bp38vDg4iHiwmyxQmEXKRBmCeQinJebIxNI5wNMzgwAABr50cH+OD+Q5+bk4eZm52zv9MWi/mvwbyI+IfHf/ryMAgQAEE7P79pf5eXWA3DHAbB1v2upWwDaVgBo3/ldM9sJoFoK0Hr5i3k4/EAenqFQyDwdHAoLC+0lYqG9MOOLPv8z4W/gi372/EAe/tt68ABxmkCZrcCjg/1xYW52rlKO58sEQjFu9+cj/seFf/2OKdHiNLFcLBWK8ViJuFAiTcd5uVKRRCHJleIS6X8y8R+W/QmTdw0ArIZPwE62B7XLbMB+7gECiw5Y0nYAQH7zLYwaC5EAEGc0Mnn3AACTv/mPQCsBAM2XpOMAALzoGFyolBdMxggAAESggSqwQQcMwRSswA6cwR28wBcCYQZEQAwkwDwQQgbkgBwKoRiWQRlUwDrYBLWwAxqgEZrhELTBMTgN5+ASXIHrcBcGYBiewhi8hgkEQcgIE2EhOogRYo7YIs4IF5mOBCJhSDSSgKQg6YgUUSLFyHKkAqlCapFdSCPyLXIUOY1cQPqQ28ggMor8irxHMZSBslED1AJ1QLmoHxqKxqBz0XQ0D12AlqJr0Rq0Hj2AtqKn0UvodXQAfYqOY4DRMQ5mjNlhXIyHRWCJWBomxxZj5Vg1Vo81Yx1YN3YVG8CeYe8IJAKLgBPsCF6EEMJsgpCQR1hMWEOoJewjtBK6CFcJg4Qxwicik6hPtCV6EvnEeGI6sZBYRqwm7iEeIZ4lXicOE1+TSCQOyZLkTgohJZAySQtJa0jbSC2kU6Q+0hBpnEwm65Btyd7kCLKArCCXkbeQD5BPkvvJw+S3FDrFiOJMCaIkUqSUEko1ZT/lBKWfMkKZoKpRzame1AiqiDqfWkltoHZQL1OHqRM0dZolzZsWQ8ukLaPV0JppZ2n3aC/pdLoJ3YMeRZfQl9Jr6Afp5+mD9HcMDYYNg8dIYigZaxl7GacYtxkvmUymBdOXmchUMNcyG5lnmA+Yb1VYKvYqfBWRyhKVOpVWlX6V56pUVXNVP9V5qgtUq1UPq15WfaZGVbNQ46kJ1Bar1akdVbupNq7OUndSj1DPUV+jvl/9gvpjDbKGhUaghkijVGO3xhmNIRbGMmXxWELWclYD6yxrmE1iW7L57Ex2Bfsbdi97TFNDc6pmrGaRZp3mcc0BDsax4PA52ZxKziHODc57LQMtPy2x1mqtZq1+rTfaetq+2mLtcu0W7eva73VwnUCdLJ31Om0693UJuja6UbqFutt1z+o+02PreekJ9cr1Dund0Uf1bfSj9Rfq79bv0R83MDQINpAZbDE4Y/DMkGPoa5hpuNHwhOGoEctoupHEaKPRSaMnuCbuh2fjNXgXPmasbxxirDTeZdxrPGFiaTLbpMSkxeS+Kc2Ua5pmutG003TMzMgs3KzYrMnsjjnVnGueYb7ZvNv8jYWlRZzFSos2i8eW2pZ8ywWWTZb3rJhWPlZ5VvVW16xJ1lzrLOtt1ldsUBtXmwybOpvLtqitm63Edptt3xTiFI8p0in1U27aMez87ArsmuwG7Tn2YfYl9m32zx3MHBId1jt0O3xydHXMdmxwvOuk4TTDqcSpw+lXZxtnoXOd8zUXpkuQyxKXdpcXU22niqdun3rLleUa7rrStdP1o5u7m9yt2W3U3cw9xX2r+00umxvJXcM970H08PdY4nHM452nm6fC85DnL152Xlle+70eT7OcJp7WMG3I28Rb4L3Le2A6Pj1l+s7pAz7GPgKfep+Hvqa+It89viN+1n6Zfgf8nvs7+sv9j/i/4XnyFvFOBWABwQHlAb2BGoGzA2sDHwSZBKUHNQWNBbsGLww+FUIMCQ1ZH3KTb8AX8hv5YzPcZyya0RXKCJ0VWhv6MMwmTB7WEY6GzwjfEH5vpvlM6cy2CIjgR2yIuB9pGZkX+X0UKSoyqi7qUbRTdHF09yzWrORZ+2e9jvGPqYy5O9tqtnJ2Z6xqbFJsY+ybuIC4qriBeIf4RfGXEnQTJAntieTE2MQ9ieNzAudsmjOc5JpUlnRjruXcorkX5unOy553PFk1WZB8OIWYEpeyP+WDIEJQLxhP5aduTR0T8oSbhU9FvqKNolGxt7hKPJLmnVaV9jjdO31D+miGT0Z1xjMJT1IreZEZkrkj801WRNberM/ZcdktOZSclJyjUg1plrQr1zC3KLdPZisrkw3keeZtyhuTh8r35CP5c/PbFWyFTNGjtFKuUA4WTC+oK3hbGFt4uEi9SFrUM99m/ur5IwuCFny9kLBQuLCz2Lh4WfHgIr9FuxYji1MXdy4xXVK6ZHhp8NJ9y2jLspb9UOJYUlXyannc8o5Sg9KlpUMrglc0lamUycturvRauWMVYZVkVe9ql9VbVn8qF5VfrHCsqK74sEa45uJXTl/VfPV5bdra3kq3yu3rSOuk626s91m/r0q9akHV0IbwDa0b8Y3lG19tSt50oXpq9Y7NtM3KzQM1YTXtW8y2rNvyoTaj9nqdf13LVv2tq7e+2Sba1r/dd3vzDoMdFTve75TsvLUreFdrvUV99W7S7oLdjxpiG7q/5n7duEd3T8Wej3ulewf2Re/ranRvbNyvv7+yCW1SNo0eSDpw5ZuAb9qb7Zp3tXBaKg7CQeXBJ9+mfHvjUOihzsPcw83fmX+39QjrSHkr0jq/dawto22gPaG97+iMo50dXh1Hvrf/fu8x42N1xzWPV56gnSg98fnkgpPjp2Snnp1OPz3Umdx590z8mWtdUV29Z0PPnj8XdO5Mt1/3yfPe549d8Lxw9CL3Ytslt0utPa49R35w/eFIr1tv62X3y+1XPK509E3rO9Hv03/6asDVc9f41y5dn3m978bsG7duJt0cuCW69fh29u0XdwruTNxdeo94r/y+2v3qB/oP6n+0/rFlwG3g+GDAYM/DWQ/vDgmHnv6U/9OH4dJHzEfVI0YjjY+dHx8bDRq98mTOk+GnsqcTz8p+Vv9563Or59/94vtLz1j82PAL+YvPv655qfNy76uprzrHI8cfvM55PfGm/K3O233vuO+638e9H5ko/ED+UPPR+mPHp9BP9z7nfP78L/eE8/sl0p8zAAAABGdBTUEAALGOfPtRkwAAACBjSFJNAAB6JQAAgIMAAPn/AACA6QAAdTAAAOpgAAA6mAAAF2+SX8VGAAAC1UlEQVR42nSQTWucZRSGr3Oe533nnZl8mDa1k5kOSampUaHaCK7cuFBwU0QXrvwRulHcu3DRvYiiP8FVQTeCkdaFQQtNNtUU25jvzEwm8/V+PcfFFLrywM3NOXBfcB8xM+5vPWAwGLx1997d2977lP8ZM1MVLVfa7U/jON4wMzzAfH2Oh389fLfXP7tZiRNMpgGxqQdAnt5CXjJXq3/QbDU3yrJEAUSEzmnnpvoIp55IPJF6vHpUHc45IvXE6ol9RLd/9o44XajWa1PA/t7e0nicXXMaIRhiggQBBERBdLoHQUUZjscvFUXxuohMAf1B7/pZr99WdZQCQY2gIE9rCBD0mQzjye6/7y23r0wBeZ69Gii8OBB5lhQMxVAzYKopUDgfnr8tyAUPcHJ8eCubDKlYSYRDzECNoFOYiCIWUBzC9CeT8eR6t9d90wNsfvPda8UgJ0oikiIjQkE9pkpQKJ2QqyGmoIK4iP5oxG/n5+97gMW9k/zySgunUD4+QswoNOCCgQmYIwBmQipGVquykGcM723e8AAsXmRw1CEQqEYxXgQRQ31MCEZRBChLDEXN8FmJFEY3z+Y9QPbGuqy2WpQqzNdnqdfquCTBxQmkY/J0QlaUDEYj8knKaHzOYeeUOKoOPUDcXMrSZpOZmVlm5i9QrSR0ul2i2efY3fiFoz/vs/bRhyyvLLO/t0/a77EwTul1uwceYKnZ+GM8GrWrlYRSjf3dx3z/1dcs3niFxsExT37dIF9/GSqere1tDg9OaLWaNBrP/+0Brlxa+nnn0c6tuBJDgMVGg08+/4yQJPx+5w6dYZ/Fi5eINWL16gs0L7c5652ytvrij2Jm/HN8MrOztfVxrVadTEbpXiVJcue1MlNNvnDObZaEB6G0+awoJmLMTtI0y9LJ9sK1tW/FzHh0eCiF+LnMV9eLPDeB0iB2Tr9E+MGJ/gTUgFJExMzUQMtg+t8AFWZM4EKAiHkAAAAASUVORK5CYII=",
        PuOffline:
          "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAACXBIWXMAAA9hAAAPYQGoP6dpAAAKTWlDQ1BQaG90b3Nob3AgSUNDIHByb2ZpbGUAAHjanVN3WJP3Fj7f92UPVkLY8LGXbIEAIiOsCMgQWaIQkgBhhBASQMWFiApWFBURnEhVxILVCkidiOKgKLhnQYqIWotVXDjuH9yntX167+3t+9f7vOec5/zOec8PgBESJpHmomoAOVKFPDrYH49PSMTJvYACFUjgBCAQ5svCZwXFAADwA3l4fnSwP/wBr28AAgBw1S4kEsfh/4O6UCZXACCRAOAiEucLAZBSAMguVMgUAMgYALBTs2QKAJQAAGx5fEIiAKoNAOz0ST4FANipk9wXANiiHKkIAI0BAJkoRyQCQLsAYFWBUiwCwMIAoKxAIi4EwK4BgFm2MkcCgL0FAHaOWJAPQGAAgJlCLMwAIDgCAEMeE80DIEwDoDDSv+CpX3CFuEgBAMDLlc2XS9IzFLiV0Bp38vDg4iHiwmyxQmEXKRBmCeQinJebIxNI5wNMzgwAABr50cH+OD+Q5+bk4eZm52zv9MWi/mvwbyI+IfHf/ryMAgQAEE7P79pf5eXWA3DHAbB1v2upWwDaVgBo3/ldM9sJoFoK0Hr5i3k4/EAenqFQyDwdHAoLC+0lYqG9MOOLPv8z4W/gi372/EAe/tt68ABxmkCZrcCjg/1xYW52rlKO58sEQjFu9+cj/seFf/2OKdHiNLFcLBWK8ViJuFAiTcd5uVKRRCHJleIS6X8y8R+W/QmTdw0ArIZPwE62B7XLbMB+7gECiw5Y0nYAQH7zLYwaC5EAEGc0Mnn3AACTv/mPQCsBAM2XpOMAALzoGFyolBdMxggAAESggSqwQQcMwRSswA6cwR28wBcCYQZEQAwkwDwQQgbkgBwKoRiWQRlUwDrYBLWwAxqgEZrhELTBMTgN5+ASXIHrcBcGYBiewhi8hgkEQcgIE2EhOogRYo7YIs4IF5mOBCJhSDSSgKQg6YgUUSLFyHKkAqlCapFdSCPyLXIUOY1cQPqQ28ggMor8irxHMZSBslED1AJ1QLmoHxqKxqBz0XQ0D12AlqJr0Rq0Hj2AtqKn0UvodXQAfYqOY4DRMQ5mjNlhXIyHRWCJWBomxxZj5Vg1Vo81Yx1YN3YVG8CeYe8IJAKLgBPsCF6EEMJsgpCQR1hMWEOoJewjtBK6CFcJg4Qxwicik6hPtCV6EvnEeGI6sZBYRqwm7iEeIZ4lXicOE1+TSCQOyZLkTgohJZAySQtJa0jbSC2kU6Q+0hBpnEwm65Btyd7kCLKArCCXkbeQD5BPkvvJw+S3FDrFiOJMCaIkUqSUEko1ZT/lBKWfMkKZoKpRzame1AiqiDqfWkltoHZQL1OHqRM0dZolzZsWQ8ukLaPV0JppZ2n3aC/pdLoJ3YMeRZfQl9Jr6Afp5+mD9HcMDYYNg8dIYigZaxl7GacYtxkvmUymBdOXmchUMNcyG5lnmA+Yb1VYKvYqfBWRyhKVOpVWlX6V56pUVXNVP9V5qgtUq1UPq15WfaZGVbNQ46kJ1Bar1akdVbupNq7OUndSj1DPUV+jvl/9gvpjDbKGhUaghkijVGO3xhmNIRbGMmXxWELWclYD6yxrmE1iW7L57Ex2Bfsbdi97TFNDc6pmrGaRZp3mcc0BDsax4PA52ZxKziHODc57LQMtPy2x1mqtZq1+rTfaetq+2mLtcu0W7eva73VwnUCdLJ31Om0693UJuja6UbqFutt1z+o+02PreekJ9cr1Dund0Uf1bfSj9Rfq79bv0R83MDQINpAZbDE4Y/DMkGPoa5hpuNHwhOGoEctoupHEaKPRSaMnuCbuh2fjNXgXPmasbxxirDTeZdxrPGFiaTLbpMSkxeS+Kc2Ua5pmutG003TMzMgs3KzYrMnsjjnVnGueYb7ZvNv8jYWlRZzFSos2i8eW2pZ8ywWWTZb3rJhWPlZ5VvVW16xJ1lzrLOtt1ldsUBtXmwybOpvLtqitm63Edptt3xTiFI8p0in1U27aMez87ArsmuwG7Tn2YfYl9m32zx3MHBId1jt0O3xydHXMdmxwvOuk4TTDqcSpw+lXZxtnoXOd8zUXpkuQyxKXdpcXU22niqdun3rLleUa7rrStdP1o5u7m9yt2W3U3cw9xX2r+00umxvJXcM970H08PdY4nHM452nm6fC85DnL152Xlle+70eT7OcJp7WMG3I28Rb4L3Le2A6Pj1l+s7pAz7GPgKfep+Hvqa+It89viN+1n6Zfgf8nvs7+sv9j/i/4XnyFvFOBWABwQHlAb2BGoGzA2sDHwSZBKUHNQWNBbsGLww+FUIMCQ1ZH3KTb8AX8hv5YzPcZyya0RXKCJ0VWhv6MMwmTB7WEY6GzwjfEH5vpvlM6cy2CIjgR2yIuB9pGZkX+X0UKSoyqi7qUbRTdHF09yzWrORZ+2e9jvGPqYy5O9tqtnJ2Z6xqbFJsY+ybuIC4qriBeIf4RfGXEnQTJAntieTE2MQ9ieNzAudsmjOc5JpUlnRjruXcorkX5unOy553PFk1WZB8OIWYEpeyP+WDIEJQLxhP5aduTR0T8oSbhU9FvqKNolGxt7hKPJLmnVaV9jjdO31D+miGT0Z1xjMJT1IreZEZkrkj801WRNberM/ZcdktOZSclJyjUg1plrQr1zC3KLdPZisrkw3keeZtyhuTh8r35CP5c/PbFWyFTNGjtFKuUA4WTC+oK3hbGFt4uEi9SFrUM99m/ur5IwuCFny9kLBQuLCz2Lh4WfHgIr9FuxYji1MXdy4xXVK6ZHhp8NJ9y2jLspb9UOJYUlXyannc8o5Sg9KlpUMrglc0lamUycturvRauWMVYZVkVe9ql9VbVn8qF5VfrHCsqK74sEa45uJXTl/VfPV5bdra3kq3yu3rSOuk626s91m/r0q9akHV0IbwDa0b8Y3lG19tSt50oXpq9Y7NtM3KzQM1YTXtW8y2rNvyoTaj9nqdf13LVv2tq7e+2Sba1r/dd3vzDoMdFTve75TsvLUreFdrvUV99W7S7oLdjxpiG7q/5n7duEd3T8Wej3ulewf2Re/ranRvbNyvv7+yCW1SNo0eSDpw5ZuAb9qb7Zp3tXBaKg7CQeXBJ9+mfHvjUOihzsPcw83fmX+39QjrSHkr0jq/dawto22gPaG97+iMo50dXh1Hvrf/fu8x42N1xzWPV56gnSg98fnkgpPjp2Snnp1OPz3Umdx590z8mWtdUV29Z0PPnj8XdO5Mt1/3yfPe549d8Lxw9CL3Ytslt0utPa49R35w/eFIr1tv62X3y+1XPK509E3rO9Hv03/6asDVc9f41y5dn3m978bsG7duJt0cuCW69fh29u0XdwruTNxdeo94r/y+2v3qB/oP6n+0/rFlwG3g+GDAYM/DWQ/vDgmHnv6U/9OH4dJHzEfVI0YjjY+dHx8bDRq98mTOk+GnsqcTz8p+Vv9563Or59/94vtLz1j82PAL+YvPv655qfNy76uprzrHI8cfvM55PfGm/K3O233vuO+638e9H5ko/ED+UPPR+mPHp9BP9z7nfP78L/eE8/sl0p8zAAAABGdBTUEAALGOfPtRkwAAACBjSFJNAAB6JQAAgIMAAPn/AACA6QAAdTAAAOpgAAA6mAAAF2+SX8VGAAACg0lEQVR42nSRT0skVxTFf+9PV1UXnRodpOm0tog2I9lMZpLtbCIkkM0Qkk+SjfkEWWSRb2C+Q1ZCsokw4G6I6JiFom7admjBqm6qy67ueu9mIS06mRw48Djce+695ykR4ejoiDzPv9rf3//VWlvyPxARrbV2q6urPwVB8EZEsABJknB6evrtcDh8GYbhxxpRSgFQVRWNRuOHdrv9xjmHBlBKcXNz89IYw4fUWt+/rbUEQUCWZd9orRfjOL4z6Pf7n04mkw1jDHPDh5xrAFpriqL4rKqqL5VSdwaj0ejZcDjsGGMQkXvOoZT6j97r9b7rdDp3BtPp9HPnnH048WHzx5Dn+ddKqacWYDAYvC7Lu/DnZ3y4tvcerTVKKWq1Gre3t8/SNH1lAXZ3d1+UZUkYhmit7wsfTnfO3Wtaa/I8R2v9vQXw3s/W1tbQWpNl2aOG+SbzfJxzAIRhyMXFxXMLEMcxWZYhIo++r1ar4b2nqqpHAXrvUUoxHo+fWIBWq6U2NjYQEZIkIY5jgiDAWotzjul0ymw2YzweU5YlRVFwfX0NMLYACwsL0ziOaTQaLC4uEkURaZpSr9c5ODjg7OyMra0tut0uV1dXpGlKkiSkafreArTb7b+LouhEUYSIcHl5yc7ODuvr64gIh4eHNJtNrLUcHx8zGAxot9u0Wq2z+Ql/nZ+fvw6CAIBms8n29jbWWvb29hiNRiwtLWGtpdvtsry8TJqmbG5u/qFEhH6/3zg5OfmxXq9PJpNJP4qimTEmjOP4Z2PMW+/9O+/9k9lsNgE+KctyWpblPysrK78pEaHX6ykgUUp9UVWVAA4ItNa/AL8bY/4EYsAppZSIaBHR3nv97wB1GTNhCWo2ugAAAABJRU5ErkJggg==",
        Mine:
          "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAAJcEhZcwAADsMAAA7DAcdvqGQAAAJpSURBVDhPjY7dS1NxGIBPIPQP5FURCuHmdGFNhjm3OZeF7UOXYRldeRF14x8Q1EiSMLHoJmLmQETb1j7yg20OxaKUREsLi3DOzXnOmUdr7UM3l5737TRO1E1zD/xufu/zPrxEDgqkUqlJJpNlKisr3wuFwhL+/2CUSuUxkUg8IxKV71dVnQGJRAJisTgkEAhqeSU3FaclFoVCjjeut6L45CkoKi4BYWkZCoWCBV7JDbc8vzjdC2zmFXicXfDgfjvotWooKioO8Epu7t5qe4J7HtzbtAG7PQwsOw7hpT4sLy8d4ZXcjLuMFWnazP6MmCFF9sI+Y4ZY0IQGvUrPKweTDPa8wbgJt4MPAZMmjPq7Z/lRfsT9xuoU2bm7G+6ANNUZ2wnck/Kj/MBA+/E09SizR3VBaq2b5r/zwwp04ccfo6OYGECWfAoY78cFxjnYs7NwlFf+j8YXu9k2k4h0fY9jHz0HDnoKnoXeQsc6iS0TdFTuIG+feOw+zOt/ueSmC5snEt6WWUStL4Ga4Q1omWTgymsamrwUqG1hUDpprPclUeEkF2VDn8v4VYIwTmGBwRedbv2A2OjdQt3YBlwYjcBZFwUqOwmqF2Gos4dAZVsFpTWAde4o1lhWqBrrcmE20DwZO9c6h2jwfoNG9yb8Dmi4QMMwBedd61DvWAP1n4BlBeRDy6CeSGP1wNdr2UCTh7l4dR7x8jsuMplAnWcLNVykYYQLvOQCTi7gDKOKe3UjDKq9MZRzF1SbZiuyAY5DuudftIbxTVejm6F0Y8xuw1iE/fcClSPE1tpXt2st/k+K/qU7EuPgEYIgiF8tmXOJr9+OYgAAAABJRU5ErkJggg==",
        CuOffline:
          "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAAJcEhZcwAADsMAAA7DAcdvqGQAAAINSURBVDhPjVM5iFpRFP2BgdRK3EXEwkIRpkphIalShRQpJbWQwspOSAotxF1R3HADHUzlChIri5BAMEGNhSBE3BcUTZFiBuJ9ue/zSGAgjgceF94959xz3+dzZ3Cl0WhSWq32Tq1Wf5PL5Vp2/zAMBoNSKpV+lslkv9EA0AAUCsVEIpE8Y5TzUKlU7/V6PTGbzUSpVIJQKAQUE7FY3GWU89DpdF+73S4cDgdoNpvgdDrBaDSCQCD4wSjnYbPZYsfjkWy3W9jv97zReDwmuFKdUc6j1Wpdr1ar03q9Bqyw2WxguVwSk8n0klEexnw+/7jb7QhWoHU6nX5hrcswmUyMi8Xilhrg9J+z2ewpa10GFKkw/h1WQKMVu74Mo9FIhBMbNDp9A1ox0Q1+GQWj/B+ZTOZNtVpd9/t9MhgMYDgcQq/Xg06nQ4rF4iEQCLy1Wq2PGf0fUCgqFAofyuUyyeVyJBaLAQqgVCoB9sDv90MoFCLpdJoEg8Gew+HQMynHtdvtq3w+/6nRaPDiZDIJ8XgcwuEwIJkX42Twer30kEQiQVwu1xL7It4AJz+v1Wokm80CTgBqgCSgKSKRyF8jauB2uwHFNBXBFK95AxS+qtfrpFKpEDQjtHnfAA+hK0SjUZJKpfgEdrv9mjdAPPL5fC/QqILiJRJuMcXpXoITrvHL4/F8x3/jncViecJxHPcHkiRurcKDy80AAAAASUVORK5CYII=",
        AudioOutOn:
          "data:image/png;base64,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",
        AudioOutOff:
          "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAACXBIWXMAAAsTAAALEwEAmpwYAAAKTWlDQ1BQaG90b3Nob3AgSUNDIHByb2ZpbGUAAHjanVN3WJP3Fj7f92UPVkLY8LGXbIEAIiOsCMgQWaIQkgBhhBASQMWFiApWFBURnEhVxILVCkidiOKgKLhnQYqIWotVXDjuH9yntX167+3t+9f7vOec5/zOec8PgBESJpHmomoAOVKFPDrYH49PSMTJvYACFUjgBCAQ5svCZwXFAADwA3l4fnSwP/wBr28AAgBw1S4kEsfh/4O6UCZXACCRAOAiEucLAZBSAMguVMgUAMgYALBTs2QKAJQAAGx5fEIiAKoNAOz0ST4FANipk9wXANiiHKkIAI0BAJkoRyQCQLsAYFWBUiwCwMIAoKxAIi4EwK4BgFm2MkcCgL0FAHaOWJAPQGAAgJlCLMwAIDgCAEMeE80DIEwDoDDSv+CpX3CFuEgBAMDLlc2XS9IzFLiV0Bp38vDg4iHiwmyxQmEXKRBmCeQinJebIxNI5wNMzgwAABr50cH+OD+Q5+bk4eZm52zv9MWi/mvwbyI+IfHf/ryMAgQAEE7P79pf5eXWA3DHAbB1v2upWwDaVgBo3/ldM9sJoFoK0Hr5i3k4/EAenqFQyDwdHAoLC+0lYqG9MOOLPv8z4W/gi372/EAe/tt68ABxmkCZrcCjg/1xYW52rlKO58sEQjFu9+cj/seFf/2OKdHiNLFcLBWK8ViJuFAiTcd5uVKRRCHJleIS6X8y8R+W/QmTdw0ArIZPwE62B7XLbMB+7gECiw5Y0nYAQH7zLYwaC5EAEGc0Mnn3AACTv/mPQCsBAM2XpOMAALzoGFyolBdMxggAAESggSqwQQcMwRSswA6cwR28wBcCYQZEQAwkwDwQQgbkgBwKoRiWQRlUwDrYBLWwAxqgEZrhELTBMTgN5+ASXIHrcBcGYBiewhi8hgkEQcgIE2EhOogRYo7YIs4IF5mOBCJhSDSSgKQg6YgUUSLFyHKkAqlCapFdSCPyLXIUOY1cQPqQ28ggMor8irxHMZSBslED1AJ1QLmoHxqKxqBz0XQ0D12AlqJr0Rq0Hj2AtqKn0UvodXQAfYqOY4DRMQ5mjNlhXIyHRWCJWBomxxZj5Vg1Vo81Yx1YN3YVG8CeYe8IJAKLgBPsCF6EEMJsgpCQR1hMWEOoJewjtBK6CFcJg4Qxwicik6hPtCV6EvnEeGI6sZBYRqwm7iEeIZ4lXicOE1+TSCQOyZLkTgohJZAySQtJa0jbSC2kU6Q+0hBpnEwm65Btyd7kCLKArCCXkbeQD5BPkvvJw+S3FDrFiOJMCaIkUqSUEko1ZT/lBKWfMkKZoKpRzame1AiqiDqfWkltoHZQL1OHqRM0dZolzZsWQ8ukLaPV0JppZ2n3aC/pdLoJ3YMeRZfQl9Jr6Afp5+mD9HcMDYYNg8dIYigZaxl7GacYtxkvmUymBdOXmchUMNcyG5lnmA+Yb1VYKvYqfBWRyhKVOpVWlX6V56pUVXNVP9V5qgtUq1UPq15WfaZGVbNQ46kJ1Bar1akdVbupNq7OUndSj1DPUV+jvl/9gvpjDbKGhUaghkijVGO3xhmNIRbGMmXxWELWclYD6yxrmE1iW7L57Ex2Bfsbdi97TFNDc6pmrGaRZp3mcc0BDsax4PA52ZxKziHODc57LQMtPy2x1mqtZq1+rTfaetq+2mLtcu0W7eva73VwnUCdLJ31Om0693UJuja6UbqFutt1z+o+02PreekJ9cr1Dund0Uf1bfSj9Rfq79bv0R83MDQINpAZbDE4Y/DMkGPoa5hpuNHwhOGoEctoupHEaKPRSaMnuCbuh2fjNXgXPmasbxxirDTeZdxrPGFiaTLbpMSkxeS+Kc2Ua5pmutG003TMzMgs3KzYrMnsjjnVnGueYb7ZvNv8jYWlRZzFSos2i8eW2pZ8ywWWTZb3rJhWPlZ5VvVW16xJ1lzrLOtt1ldsUBtXmwybOpvLtqitm63Edptt3xTiFI8p0in1U27aMez87ArsmuwG7Tn2YfYl9m32zx3MHBId1jt0O3xydHXMdmxwvOuk4TTDqcSpw+lXZxtnoXOd8zUXpkuQyxKXdpcXU22niqdun3rLleUa7rrStdP1o5u7m9yt2W3U3cw9xX2r+00umxvJXcM970H08PdY4nHM452nm6fC85DnL152Xlle+70eT7OcJp7WMG3I28Rb4L3Le2A6Pj1l+s7pAz7GPgKfep+Hvqa+It89viN+1n6Zfgf8nvs7+sv9j/i/4XnyFvFOBWABwQHlAb2BGoGzA2sDHwSZBKUHNQWNBbsGLww+FUIMCQ1ZH3KTb8AX8hv5YzPcZyya0RXKCJ0VWhv6MMwmTB7WEY6GzwjfEH5vpvlM6cy2CIjgR2yIuB9pGZkX+X0UKSoyqi7qUbRTdHF09yzWrORZ+2e9jvGPqYy5O9tqtnJ2Z6xqbFJsY+ybuIC4qriBeIf4RfGXEnQTJAntieTE2MQ9ieNzAudsmjOc5JpUlnRjruXcorkX5unOy553PFk1WZB8OIWYEpeyP+WDIEJQLxhP5aduTR0T8oSbhU9FvqKNolGxt7hKPJLmnVaV9jjdO31D+miGT0Z1xjMJT1IreZEZkrkj801WRNberM/ZcdktOZSclJyjUg1plrQr1zC3KLdPZisrkw3keeZtyhuTh8r35CP5c/PbFWyFTNGjtFKuUA4WTC+oK3hbGFt4uEi9SFrUM99m/ur5IwuCFny9kLBQuLCz2Lh4WfHgIr9FuxYji1MXdy4xXVK6ZHhp8NJ9y2jLspb9UOJYUlXyannc8o5Sg9KlpUMrglc0lamUycturvRauWMVYZVkVe9ql9VbVn8qF5VfrHCsqK74sEa45uJXTl/VfPV5bdra3kq3yu3rSOuk626s91m/r0q9akHV0IbwDa0b8Y3lG19tSt50oXpq9Y7NtM3KzQM1YTXtW8y2rNvyoTaj9nqdf13LVv2tq7e+2Sba1r/dd3vzDoMdFTve75TsvLUreFdrvUV99W7S7oLdjxpiG7q/5n7duEd3T8Wej3ulewf2Re/ranRvbNyvv7+yCW1SNo0eSDpw5ZuAb9qb7Zp3tXBaKg7CQeXBJ9+mfHvjUOihzsPcw83fmX+39QjrSHkr0jq/dawto22gPaG97+iMo50dXh1Hvrf/fu8x42N1xzWPV56gnSg98fnkgpPjp2Snnp1OPz3Umdx590z8mWtdUV29Z0PPnj8XdO5Mt1/3yfPe549d8Lxw9CL3Ytslt0utPa49R35w/eFIr1tv62X3y+1XPK509E3rO9Hv03/6asDVc9f41y5dn3m978bsG7duJt0cuCW69fh29u0XdwruTNxdeo94r/y+2v3qB/oP6n+0/rFlwG3g+GDAYM/DWQ/vDgmHnv6U/9OH4dJHzEfVI0YjjY+dHx8bDRq98mTOk+GnsqcTz8p+Vv9563Or59/94vtLz1j82PAL+YvPv655qfNy76uprzrHI8cfvM55PfGm/K3O233vuO+638e9H5ko/ED+UPPR+mPHp9BP9z7nfP78L/eE8/sl0p8zAAAABGdBTUEAALGOfPtRkwAAACBjSFJNAAB6JQAAgIMAAPn/AACA6QAAdTAAAOpgAAA6mAAAF2+SX8VGAAADL0lEQVR42kyRTWsrZRiG7+eZr7wzyYydkqRpkybZhOKiSpfiQg5oF7oQpHBwIbi0+A8EN0U5yNm6PODSvcLZit24kFoOpTbQj0yIZYY004RM0k5m5n1dWbx/wMXFddNgMECSJJjP55jNZsjzHK7rIk1TOI5DURQ9D8PwC2beq1Qq82q1+tq27Ree5/3DzNDxvzEziAhEhKIovCRJXo3H48+WyyXyPEeSJLU4jr9ut9sf27b9KRG90YkImqZBSmnOZrNnhmG8tVwuL2zbPhqPx59EUYQ0TcHMYGasViv0+/0uM//YaDQ+0IuiQBzH715cXLx6fHzcS5IEzWZTNptNPj09BTMjz3MopWBZFpIkgeM4uLy8fL9arX7Ok8mkfHV19dNisdgrl8uwLAu7u7t8fHyMIAgQRRGyLINhGJBSQtM0pGkKKSUmk8k+R1H01Ww2e0dKieVyiZ2dHQwGA0ynU/i+DyICM6MoCgCAUgpZloGZAeBt7vf7H2ZZhnq9DiklXNfF9fU1Dg8PcXBwAGbG9vY2iOgJwMyQUiJN0xqHYdglIpycnGA0GiHPczAzut0utra2sL+/j0ajAcMwnt5SSmE+n2M6nbJ2dHT0nJlbvV4PRVFgsVhgOBwiSRLEcYxWq4XRaIRer4dWqwXP8xCGIQCgVquNdNd1f18sFu91Oh3ouo4gCP4LBCEEVqsVhBAQQkDXdei6Dsuy0G630el0hqzr+stSqXQaBAGEEPB9H3EcIwxD3N7e4ubmBmtra0/q0+kUtm2j1WqhXC7/yUqpieM4zyzL+sYwjHNN02DbNkajEYbDIcIwxP39PUzThJQSQRBgfX0duq7LPM9/1QFASnmvadr3lmX9YprmX/V6XXt4eECWZSAinJ2dQUqJ8XiMoihQrVZRqVReF0XxBxMRLMuC53kolUp/u677plarQUoJpRSICFJKnJ+fYzKZYHNzE0KITCn1UtM0sGmacBwHpVIJRJTbtv2D7/vY2NiAaZogIqxWK2RZBtu2YRgGKpXKt5qm/cbM0JVST4EAQAjxs+/7HwH4kohwd3eHLMuQ5zkAqHq9/p0Q4gURQSmFfwcARRWDhWf50LkAAAAASUVORK5CYII=",
        AudioInOn:
          "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAACXBIWXMAAAsTAAALEwEAmpwYAAAKTWlDQ1BQaG90b3Nob3AgSUNDIHByb2ZpbGUAAHjanVN3WJP3Fj7f92UPVkLY8LGXbIEAIiOsCMgQWaIQkgBhhBASQMWFiApWFBURnEhVxILVCkidiOKgKLhnQYqIWotVXDjuH9yntX167+3t+9f7vOec5/zOec8PgBESJpHmomoAOVKFPDrYH49PSMTJvYACFUjgBCAQ5svCZwXFAADwA3l4fnSwP/wBr28AAgBw1S4kEsfh/4O6UCZXACCRAOAiEucLAZBSAMguVMgUAMgYALBTs2QKAJQAAGx5fEIiAKoNAOz0ST4FANipk9wXANiiHKkIAI0BAJkoRyQCQLsAYFWBUiwCwMIAoKxAIi4EwK4BgFm2MkcCgL0FAHaOWJAPQGAAgJlCLMwAIDgCAEMeE80DIEwDoDDSv+CpX3CFuEgBAMDLlc2XS9IzFLiV0Bp38vDg4iHiwmyxQmEXKRBmCeQinJebIxNI5wNMzgwAABr50cH+OD+Q5+bk4eZm52zv9MWi/mvwbyI+IfHf/ryMAgQAEE7P79pf5eXWA3DHAbB1v2upWwDaVgBo3/ldM9sJoFoK0Hr5i3k4/EAenqFQyDwdHAoLC+0lYqG9MOOLPv8z4W/gi372/EAe/tt68ABxmkCZrcCjg/1xYW52rlKO58sEQjFu9+cj/seFf/2OKdHiNLFcLBWK8ViJuFAiTcd5uVKRRCHJleIS6X8y8R+W/QmTdw0ArIZPwE62B7XLbMB+7gECiw5Y0nYAQH7zLYwaC5EAEGc0Mnn3AACTv/mPQCsBAM2XpOMAALzoGFyolBdMxggAAESggSqwQQcMwRSswA6cwR28wBcCYQZEQAwkwDwQQgbkgBwKoRiWQRlUwDrYBLWwAxqgEZrhELTBMTgN5+ASXIHrcBcGYBiewhi8hgkEQcgIE2EhOogRYo7YIs4IF5mOBCJhSDSSgKQg6YgUUSLFyHKkAqlCapFdSCPyLXIUOY1cQPqQ28ggMor8irxHMZSBslED1AJ1QLmoHxqKxqBz0XQ0D12AlqJr0Rq0Hj2AtqKn0UvodXQAfYqOY4DRMQ5mjNlhXIyHRWCJWBomxxZj5Vg1Vo81Yx1YN3YVG8CeYe8IJAKLgBPsCF6EEMJsgpCQR1hMWEOoJewjtBK6CFcJg4Qxwicik6hPtCV6EvnEeGI6sZBYRqwm7iEeIZ4lXicOE1+TSCQOyZLkTgohJZAySQtJa0jbSC2kU6Q+0hBpnEwm65Btyd7kCLKArCCXkbeQD5BPkvvJw+S3FDrFiOJMCaIkUqSUEko1ZT/lBKWfMkKZoKpRzame1AiqiDqfWkltoHZQL1OHqRM0dZolzZsWQ8ukLaPV0JppZ2n3aC/pdLoJ3YMeRZfQl9Jr6Afp5+mD9HcMDYYNg8dIYigZaxl7GacYtxkvmUymBdOXmchUMNcyG5lnmA+Yb1VYKvYqfBWRyhKVOpVWlX6V56pUVXNVP9V5qgtUq1UPq15WfaZGVbNQ46kJ1Bar1akdVbupNq7OUndSj1DPUV+jvl/9gvpjDbKGhUaghkijVGO3xhmNIRbGMmXxWELWclYD6yxrmE1iW7L57Ex2Bfsbdi97TFNDc6pmrGaRZp3mcc0BDsax4PA52ZxKziHODc57LQMtPy2x1mqtZq1+rTfaetq+2mLtcu0W7eva73VwnUCdLJ31Om0693UJuja6UbqFutt1z+o+02PreekJ9cr1Dund0Uf1bfSj9Rfq79bv0R83MDQINpAZbDE4Y/DMkGPoa5hpuNHwhOGoEctoupHEaKPRSaMnuCbuh2fjNXgXPmasbxxirDTeZdxrPGFiaTLbpMSkxeS+Kc2Ua5pmutG003TMzMgs3KzYrMnsjjnVnGueYb7ZvNv8jYWlRZzFSos2i8eW2pZ8ywWWTZb3rJhWPlZ5VvVW16xJ1lzrLOtt1ldsUBtXmwybOpvLtqitm63Edptt3xTiFI8p0in1U27aMez87ArsmuwG7Tn2YfYl9m32zx3MHBId1jt0O3xydHXMdmxwvOuk4TTDqcSpw+lXZxtnoXOd8zUXpkuQyxKXdpcXU22niqdun3rLleUa7rrStdP1o5u7m9yt2W3U3cw9xX2r+00umxvJXcM970H08PdY4nHM452nm6fC85DnL152Xlle+70eT7OcJp7WMG3I28Rb4L3Le2A6Pj1l+s7pAz7GPgKfep+Hvqa+It89viN+1n6Zfgf8nvs7+sv9j/i/4XnyFvFOBWABwQHlAb2BGoGzA2sDHwSZBKUHNQWNBbsGLww+FUIMCQ1ZH3KTb8AX8hv5YzPcZyya0RXKCJ0VWhv6MMwmTB7WEY6GzwjfEH5vpvlM6cy2CIjgR2yIuB9pGZkX+X0UKSoyqi7qUbRTdHF09yzWrORZ+2e9jvGPqYy5O9tqtnJ2Z6xqbFJsY+ybuIC4qriBeIf4RfGXEnQTJAntieTE2MQ9ieNzAudsmjOc5JpUlnRjruXcorkX5unOy553PFk1WZB8OIWYEpeyP+WDIEJQLxhP5aduTR0T8oSbhU9FvqKNolGxt7hKPJLmnVaV9jjdO31D+miGT0Z1xjMJT1IreZEZkrkj801WRNberM/ZcdktOZSclJyjUg1plrQr1zC3KLdPZisrkw3keeZtyhuTh8r35CP5c/PbFWyFTNGjtFKuUA4WTC+oK3hbGFt4uEi9SFrUM99m/ur5IwuCFny9kLBQuLCz2Lh4WfHgIr9FuxYji1MXdy4xXVK6ZHhp8NJ9y2jLspb9UOJYUlXyannc8o5Sg9KlpUMrglc0lamUycturvRauWMVYZVkVe9ql9VbVn8qF5VfrHCsqK74sEa45uJXTl/VfPV5bdra3kq3yu3rSOuk626s91m/r0q9akHV0IbwDa0b8Y3lG19tSt50oXpq9Y7NtM3KzQM1YTXtW8y2rNvyoTaj9nqdf13LVv2tq7e+2Sba1r/dd3vzDoMdFTve75TsvLUreFdrvUV99W7S7oLdjxpiG7q/5n7duEd3T8Wej3ulewf2Re/ranRvbNyvv7+yCW1SNo0eSDpw5ZuAb9qb7Zp3tXBaKg7CQeXBJ9+mfHvjUOihzsPcw83fmX+39QjrSHkr0jq/dawto22gPaG97+iMo50dXh1Hvrf/fu8x42N1xzWPV56gnSg98fnkgpPjp2Snnp1OPz3Umdx590z8mWtdUV29Z0PPnj8XdO5Mt1/3yfPe549d8Lxw9CL3Ytslt0utPa49R35w/eFIr1tv62X3y+1XPK509E3rO9Hv03/6asDVc9f41y5dn3m978bsG7duJt0cuCW69fh29u0XdwruTNxdeo94r/y+2v3qB/oP6n+0/rFlwG3g+GDAYM/DWQ/vDgmHnv6U/9OH4dJHzEfVI0YjjY+dHx8bDRq98mTOk+GnsqcTz8p+Vv9563Or59/94vtLz1j82PAL+YvPv655qfNy76uprzrHI8cfvM55PfGm/K3O233vuO+638e9H5ko/ED+UPPR+mPHp9BP9z7nfP78L/eE8/sl0p8zAAAABGdBTUEAALGOfPtRkwAAACBjSFJNAAB6JQAAgIMAAPn/AACA6QAAdTAAAOpgAAA6mAAAF2+SX8VGAAAC/UlEQVR42nxSS2hTaRg9/73/fZjcvHrTmMzkSptWUzuKrZbB0UFFQUFRoeL7sRJxIc4M6EKsq65kGMWFOrhooZ3FLGaGEVFQfKGgETS1WqwWsdrYVNvGpGl7c9/XhTYqUj848MHHOd/5Dh+JrzuBqVI1EysWVeHf41vQeakbHoHFplXzyvM7ve+WtnQ+PWWxpNIrCB2pjkOtFN8ohpBy//etwZ3HOp/+SRhGCgREePxiS7xpPcdMRyaEID+uAgAupLJrfzvX85dpORJlCDTVQCFXRHjOkj3TCgg8xfOXWei6jsHRks+yAZ4yME0btu2iUNRRLKiT056gG1bj+Zs9LQN9D+95nFw8zC2FhhgoTIxNGAhK3GjrjroDXwkIHIvr9wc2/PH/q/Z3tLbi0pWzzdVCBjV1eQyKzcNDJV9kbtzbfXLfD7saEsEeomw8XSY7jougxP9ksMGboUicdwgH9c1jFFKnEItUtLW3tR1WSSiZ/N7zJChxYwDA2uooSkPdH/EQ+YFHNhWl3WJlvW9l0gUzQ8arcT+Gx0xvXa3SvrJJ6QtIM/SppaxZHMQnZGPVMjrmznQWvC15kDFiGM3lYRReAo5++faDZxfyY6q++uf5ZdefZ1DR2Nh4I5FIJF88f/ZYnzhzzbH3/uIYE5aj5X93CX/UchyAYb/IjBJCoCgKvF7vQZ/P9106nT7T399/ZHZNdVM9c/dXOxBZO2RNXn2UToGYE/gvdwvG6zuIRqNQFOWDiizL8xRF2UEpjUwpz4yEN2zfttX1ilz4m99KKa0pFouhTCbzj2VZRQCVDOdJsGJoPsNSuGJ4GeE8sWk/VhTFZk3TugAYgly7xlu1bD8rRWfL/Lh/cXSEuTFS75isL29N5nrG+y62lrJd177IwHVdg+f5BriOG5y18IivZkmtaTOwXBVPSjKonGR4MSTLQf9yQ5qk6WzXdQBu2QEAnhAShutylBd/DFc1bJaiyXrBHwuwgp9nnJJpqyMFbbi3a6A3dU7T9NufO3g/AH84OU0/uEVjAAAAAElFTkSuQmCC",
        AudioInOff:
          "data:image/png;base64,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"
    };

    function showUser(szUser) {
        var tempuser = null;
        var tempUserEle = null;
        for (iindex in szUser) {
            tempuser = szUser[iindex];
            tempUserEle = getUserEleFromCache(tempuser);
            tempUserEle.fill(tempuser);
        }
    }

    function getUserEleFromCache(User) {
        var tempusercache = null;
        for (iindex in uiglobel.userlist) {
            tempusercache = uiglobel.userlist[iindex];
            if (tempusercache.check(User)) {
                return tempusercache;
            }
        }

        var userEle = new UserEle(User.id);
        uiglobel.userContentUl.appendChild(userEle.getEle());
        uiglobel.userlist.push(userEle);
        return userEle;
    }

    function onApplySpeakHasResult(options, response) {
        if (checkJswResult(response.emms.code)) {
            uiglobel.apply();
        }
    }

    function applySpeak() {
        var conf = gparams.getDefaultConf();
        if (conf) {
            var params = {
                callback: onApplySpeakHasResult,
                tag: null
            };
            var rc = conf.swApplyForSpeak(params);

            if (!checkJswResult(rc)) {
            }
        } else {
            infolog("");
        }
    }

    function onApplyEndSpeakHasResult(options, response) {
        if (checkJswResult(response.emms.code)) {
            uiglobel.disapply();
        }
    }

    function applyStopSpeak() {
        var conf = gparams.getDefaultConf();
        if (conf) {
            var params = {
                callback: onApplyEndSpeakHasResult,
                tag: null
            };
            var rc = conf.swApplyForEndSpeak(params);
            if (!checkJswResult(rc)) {
            }
        } else {
            infolog("");
        }
    }

    function onNotifyConfStatusChange(sender, event, msg) {
        switch (event) {
            case "notifyapplyforstartspeak":
            case "notifyapplyforendspeak":
            case "notifyparticipatormodify":
            case "notifyparticipartorreturn":
            case "notifyparticipartorleave":
                getDefaultUserlistAndShow();
                break;
        }
    }

    var groupCallDebug = {};
    (function () {
        groupCallDebug.onInitOk = function () { };
    })(groupCallDebug);
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
				console.log(this.cmpName + ": " + text);
				if(obj){
					console.log(obj);
				}
			};

			console.log("jsw plugin common js load ok!");
		}
	}else{
		console.log("jsw plugin common js load faild!");
	}
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
				console.log(this.cmpName + ": " + text);
				if(obj){
					console.log(obj);
				}
			};

			console.log("jsw plugin common js load ok!");
		}
	}else{
		console.log("jsw plugin common js load faild!");
	}
})();
