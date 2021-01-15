module.exports = {
    title: '論文筆記',
    description: '電腦視覺與深度學習論文筆記',
    base: '/paper-notes/',
    head: [
      ['meta', { name: "viewport", content: "width=device-width,user-scaleble=0,initial-scale=1.0,maximum-scale=1.0" }],
    ],
    locales: {
      '/': {
        lang: 'zh-TW',
      }
    },
    repo: 'https://github.com/aquastripe/paper-notes',
    themeConfig: {
      sidebarDepth: 0,
      sidebar: [
        ['/', 'Preface'],
        {
          title: '2017',
          collapsable: false,
          children: [
              '2017/attention-is-all-you-need'
          ],
        }
      ],
      nav: [
        { text: 'Home', link: '/' },
        { text: 'Github', link: 'https://github.com/aquastripe/paper-notes' }
      ],
      tags: [
        '2017',
        'NeurIPS',
        'NLP',
        'Attention',
      ]
    },
    markdown: {
      lineNumbers: true
    }
  }
  