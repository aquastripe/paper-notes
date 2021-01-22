module.exports = {
    title: '論文筆記',
    description: '電腦視覺與深度學習論文筆記',
    base: '/paper-notes/',
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
          title: '2021',
          collapsable: false,
          children: [
            '2021/an-image-is-worth-16x16-words-transformers-for-image-recognition-at-scale'
          ],
        },
        {
          title: '2019',
          collapsable: false,
          children: [
              '2019/auto-augment-learning-augmentation-strategies-from-data'
          ],
        },
        // {
        //   title: '2017',
        //   collapsable: false,
        //   children: [
        //     '2017/attention-is-all-you-need',
        //   ],
        // }
      ],
      nav: [
        { text: 'Home', link: '/' },
        { text: 'Github', link: 'https://github.com/aquastripe/paper-notes' }
      ],
      tags: [
        // years
        '2017',
        '2018',
        '2019',
        '2020',
        '2021',
        
        // conference, journel
        'CVPR',
        'NeurIPS',
        'ICLR',

        // topic
        'NLP',
        'Attention',
        'Data Augmentation',
        'Image Recognition',
        'Vision Transformer',
        'ViT',
      ]
    },
    markdown: {
      lineNumbers: true,
    },
    plugins: [
      'mathjax'
    ]
  }
  