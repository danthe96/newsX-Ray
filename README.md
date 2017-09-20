# News X-Ray

This is the backend (frontend [here](https://github.com/danthe96/newsX-Ray-frontend)) for News X-Ray, a Chrome extension identifying agency reports that were used for news articles. That way, you know which information is well-founded and what might be controversial. Through analysis we can show biases differ between news sources.
News X-Ray was created by [Daniel Thevessen](https://github.com/danthe96), [Carl Goedecken](https://github.com/MasterCarl) and [Nils Strelow](https://github.com/nstrelow) at [HackZurich 2017](hackzurich.com) within just under two days. 

## Inspiration

We all read news articles in some shape or form almost everyday. But in times of fake news and clickbait, how can you know who to trust? No matter what source you're reading, news should be transparent. That's why we want to show what information is behind a news article, enabling us to show what the news source might have added as well. That way, you can better identify the biases of individual news sources, and check if this is a trend for this source.

## What it does

We have developed a browser extension that looks at the article you are currently reading, and identifies a related report from a news agency such as Reuters where the basic information and facts come from. A backend finds the parts of the article that were derived from this report - even if the author has rewritten them completely - through Natural Language Processing. The extension highlights the similarities and saves sentiment trends to show how news sources differ. 

## How we built it

We identify related Reuter reports by finding keywords with IBM Bluemix and searching recent news history through the Reuters API. Once it has found one, the extensions communicates with a Python Flask backend, which runs semantic sentence analysis on both articles to find similar meaning. We do so by generating sentence vectors with Facebook's fasttext and a model pre-trained on the English language. The extension takes this information to highlight things, and run 

## Challenges we ran into

Semantic sentence similarity is far from easy and still an open topic of research! There is no clear way of extracting meaning from whole sentences, so we had to try out many things. We even got to the limit, of what free hosting vouchers could offer. With a model larger than 10 GB, we were lucky to have a Macbook with 16 GB of RAM.

## Accomplishments that we're proud of

We're proud to have found a solution that not only works, but can also provide great insight on real articles. And, despite getting little sleep, we have all made it through HackZurich in one piece!

## What we learned

There is still so much AI can do for us, that we haven't even begun to think about!

## What's next for News X-Ray

Decomposing articles using data is great, but the data generated from that itself can be used as well. It would be great to collect more data and explore what we can do with that.
