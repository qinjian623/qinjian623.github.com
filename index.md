---
layout: page
title: 不用怀疑,此乃首页
---
{% include JB/setup %}

    
刚换过来,首页简单奔放点:

<p>favicon也咩有...</p>
<p>得先弄个可爱点的404......</p>
<p>列表清单</p>
<ul class="posts">
  {% for post in site.posts %}
    <li><span>{{ post.date | date_to_string }}</span> &raquo; <a href="{{ BASE_PATH }}{{ post.url }}">{{ post.title }}</a></li>
  {% endfor %}
</ul>
