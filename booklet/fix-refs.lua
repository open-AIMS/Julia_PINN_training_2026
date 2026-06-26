-- Booklet-only fix. The references appendix uses hand-written ::: {#ref-KEY}
-- divs (styled bibliography). Pandoc turns any div whose id starts with "ref-"
-- into a \bibitem in LaTeX, which needs a thebibliography environment that
-- isn't there -> "Lonely \item". For LaTeX output, drop the id so each entry
-- renders as an ordinary text block. (HTML/site output is untouched.)
function Div(el)
  if FORMAT:match('latex') and el.identifier:match('^ref%-') then
    el.identifier = ''
  end
  return el
end
