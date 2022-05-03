import React from "react";
import { graphql } from "gatsby";
import { Header } from "../pages/header";

export default function DocPage({ pageContext }) {
  console.log(pageContext);

  return (
    <div className="api-wrapper">
      <div className="api-title">Module: {pageContext.name}</div>

      <div className="api-body">
        <RenderNodes nodes={pageContext.docstring.sections.body} />
      </div>

      <div className="api-classes">
        <div className="api-subtitle">Classes</div>
        {Object.entries(pageContext.classes).map(([name, class_]) => {
          return (
            <p className="api-toc-item">
              <a
                className="api-toc-item-link"
                href={`${pageContext.name}/${name}`}
              >
                class {name}:
              </a>
              {" " + class_.sections.body[0].content}
            </p>
          );
        })}
      </div>

      {pageContext.functions.length > 0 ? (
        <div className="api-functions">
          <div className="api-subtitle">Functions</div>
          {Object.entries(pageContext.functions).map(([name, class_]) => {
            return (
              <p className="api-toc-item">
                <a
                  className="api-toc-item-link"
                  href={`docs/${name}/${class_.name}`}
                >
                  {name}(...):
                </a>
                {class_.sections.body[0].content}
              </p>
            );
          })}
        </div>
      ) : null}
    </div>
  );
}

function RenderNodes({ nodes }) {
  return nodes.map((node, i) => {
    if (node.type === "Text") {
      return (
        <span className={node.type} key={i}>
          {node.content}
        </span>
      );
    }

    switch (node.type) {
      case "section":
        return (
          <div className="api-section" key={i}>
            <RenderNodes nodes={node.children} />
          </div>
        );
      case "title":
        return (
          <div className="title" key={i}>
            <RenderNodes nodes={node.children} />
          </div>
        );
      case "paragraph":
        return (
          <p className="paragraph">
            <RenderNodes nodes={node.children} />
          </p>
        );
      case "title_reference":
        return (
          <span className="title_reference">
            <RenderNodes nodes={node.children} />
          </span>
        );
      case "definition_list":
        return (
          <ul className={node.type}>
            <RenderNodes nodes={node.children} />
          </ul>
        );
      case "definition_list_item":
        return (
          <li className={node.type}>
            <RenderNodes nodes={node.children} />
          </li>
        );
      default:
        return (
          <div className={node.type}>
            <RenderNodes nodes={node.children} />
          </div>
        );
    }
  });
}
