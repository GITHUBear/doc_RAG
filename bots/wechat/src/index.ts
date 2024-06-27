#!/usr/bin/env -S node --no-warnings --loader ts-node/esm
/**
 * Wechaty - Conversational RPA SDK for Chatbot Makers.
 *  - https://github.com/wechaty/wechaty
 */
// https://stackoverflow.com/a/42817956/1123955
// https://github.com/motdotla/dotenv/issues/89#issuecomment-587753552
import 'dotenv/config.js'

import {
  Contact,
  Message,
  ScanStatus,
  WechatyBuilder,
  log,
} from 'wechaty'
import * as PUPPET from 'wechaty-puppet'

import qrcodeTerminal from 'qrcode-terminal'

type ChatCompletionResponse = {
  message_id: string
  conversation_id: string
  mode: string
  answer: string
  metadata: any
  created_at: number
}

function onScan(qrcode: string, status: ScanStatus) {
  if (status === ScanStatus.Waiting || status === ScanStatus.Timeout) {
    const qrcodeImageUrl = [
      'https://wechaty.js.org/qrcode/',
      encodeURIComponent(qrcode),
    ].join('')
    log.info('OceanBase Bot', 'onScan: %s(%s) - %s', ScanStatus[status], status, qrcodeImageUrl)

    qrcodeTerminal.generate(qrcode, { small: true })  // show qrcode on console

  } else {
    log.info('OceanBase Bot', 'onScan: %s(%s)', ScanStatus[status], status)
  }
}

function onLogin(user: Contact) {
  log.info('OceanBase Bot', '%s login', user)
}

function onLogout(user: Contact) {
  log.info('OceanBase Bot', '%s logout', user)
}

const QUERY_URL = process.env["QUERY_URL"]
const API_KEY = process.env["API_KEY"]
const DEBUG_TALKER = process.env["DEBUG_TALKER"]
const BOT_NAME = process.env["BOT_NAME"]

async function queryDify(msg: Message) {
  const res = await fetch(`${QUERY_URL}/v1/chat-messages`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Authorization": `Bearer ${API_KEY}`
    },
    body: JSON.stringify({
      inputs: {},
      files: [],
      response_mode: "blocking",
      user: msg.conversation().id,
      query: msg.text(),
      conversation_id: null,
      auto_generate_name: true,
    }),
  })
  if (res.status !== 200) {
    msg.say("对不起，因为服务出现问题，我暂时无法回答您的问题")
    return
  } else {
    const resp = await res.json() as ChatCompletionResponse
    console.log(resp)
    await msg.say(`你好, ${msg.talker()?.name() || "用户"}\n` + resp.answer)
  }
}

async function queryCustom(msg: Message) {
  const res = await fetch(`${QUERY_URL}/chat`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      input: msg.text(),
    }),
  })
  if (res.status !== 200) {
    msg.say("对不起，因为服务出现问题，我暂时无法回答您的问题")
    return
  } else {
    const resp = await res.json() as ChatCompletionResponse
    console.log(resp)
    await msg.say(`你好, ${msg.talker()?.name() || "用户"}\n` + resp.answer)
  }
}

async function onMessage(msg: Message) {
  const room = msg.room()
  const talker = msg.talker()
  log.info('OceanBase Bot', msg.toString())
  if (msg.type() !== PUPPET.types.Message.Text) {
    return
  }
  if (!room && talker.name() !== DEBUG_TALKER) {
    return
  }
  if (!!room && !await msg.mentionSelf()) {
    return
  }

  const question = msg.text().replace("@" + BOT_NAME, "").trim()

  await msg.say(`@${talker.name()}，我正在思考如何回答您的提问: \n` + question)
  if (question.includes("+dify")) {
    await queryDify(msg)
  } else {
    await queryCustom(msg)
  }
}

const bot = WechatyBuilder.build({
  name: 'oceanbase doc bot',
  /**
   * You can specific `puppet` and `puppetOptions` here with hard coding:
   *
  puppet: 'wechaty-puppet-wechat',
  puppetOptions: {
    uos: true,
  },
   */
  /**
   * How to set Wechaty Puppet Provider:
   *
   *  1. Specify a `puppet` option when instantiating Wechaty. (like `{ puppet: 'wechaty-puppet-whatsapp' }`, see below)
   *  1. Set the `WECHATY_PUPPET` environment variable to the puppet NPM module name. (like `wechaty-puppet-whatsapp`)
   *
   * You can use the following providers locally:
   *  - wechaty-puppet-wechat (web protocol, no token required)
   *  - wechaty-puppet-whatsapp (web protocol, no token required)
   *  - wechaty-puppet-padlocal (pad protocol, token required)
   *  - etc. see: <https://wechaty.js.org/docs/puppet-providers/>
   */
  // puppet: 'wechaty-puppet-whatsapp'

  /**
   * You can use wechaty puppet provider 'wechaty-puppet-service'
   *   which can connect to remote Wechaty Puppet Services
   *   for using more powerful protocol.
   * Learn more about services (and TOKEN) from https://wechaty.js.org/docs/puppet-services/
   */
  // puppet: 'wechaty-puppet-service'
  // puppetOptions: {
  //   token: 'xxx',
  // }
})

bot.on('scan', onScan)
bot.on('login', onLogin)
bot.on('logout', onLogout)
bot.on('message', onMessage)

bot.start()
  .then(() => log.info('OceanBase Bot', 'Bot Started.'))
  .catch(e => log.error('OceanBase Bot', e))
