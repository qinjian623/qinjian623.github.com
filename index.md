---
layout: page
title: 不用怀疑,此乃首页
---
{% include JB/setup %}

    
<p>首页简单奔放点:</p>
RSS:
<a href="http://fusion.google.com/add?source=atgs&feedurl=blog.qinjian.me/atom.xml">
<img src="http://buttons.googlesyndication.com/fusion/add.gif" alt="Add to Google Reader"/>
</a>

<p>列表清单:</p>
<ul class="posts">
  {% for post in site.posts %}
    <li><span>{{ post.date | date_to_string }}</span> &raquo; <a href="{{ BASE_PATH }}{{ post.url }}">{{ post.title }}</a></li>
  {% endfor %}
</ul>
