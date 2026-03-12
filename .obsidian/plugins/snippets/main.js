'use strict';

var obsidian = require('obsidian');

/*! *****************************************************************************
Copyright (c) Microsoft Corporation.

Permission to use, copy, modify, and/or distribute this software for any
purpose with or without fee is hereby granted.

THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH
REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY
AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT,
INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR
OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
PERFORMANCE OF THIS SOFTWARE.
***************************************************************************** */
/* global Reflect, Promise */

var extendStatics = function(d, b) {
    extendStatics = Object.setPrototypeOf ||
        ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
        function (d, b) { for (var p in b) if (Object.prototype.hasOwnProperty.call(b, p)) d[p] = b[p]; };
    return extendStatics(d, b);
};

function __extends(d, b) {
    extendStatics(d, b);
    function __() { this.constructor = d; }
    d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
}

function __awaiter(thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
}

function __generator(thisArg, body) {
    var _ = { label: 0, sent: function() { if (t[0] & 1) throw t[1]; return t[1]; }, trys: [], ops: [] }, f, y, t, g;
    return g = { next: verb(0), "throw": verb(1), "return": verb(2) }, typeof Symbol === "function" && (g[Symbol.iterator] = function() { return this; }), g;
    function verb(n) { return function (v) { return step([n, v]); }; }
    function step(op) {
        if (f) throw new TypeError("Generator is already executing.");
        while (_) try {
            if (f = 1, y && (t = op[0] & 2 ? y["return"] : op[0] ? y["throw"] || ((t = y["return"]) && t.call(y), 0) : y.next) && !(t = t.call(y, op[1])).done) return t;
            if (y = 0, t) op = [op[0] & 2, t.value];
            switch (op[0]) {
                case 0: case 1: t = op; break;
                case 4: _.label++; return { value: op[1], done: false };
                case 5: _.label++; y = op[1]; op = [0]; continue;
                case 7: op = _.ops.pop(); _.trys.pop(); continue;
                default:
                    if (!(t = _.trys, t = t.length > 0 && t[t.length - 1]) && (op[0] === 6 || op[0] === 2)) { _ = 0; continue; }
                    if (op[0] === 3 && (!t || (op[1] > t[0] && op[1] < t[3]))) { _.label = op[1]; break; }
                    if (op[0] === 6 && _.label < t[1]) { _.label = t[1]; t = op; break; }
                    if (t && _.label < t[2]) { _.label = t[2]; _.ops.push(op); break; }
                    if (t[2]) _.ops.pop();
                    _.trys.pop(); continue;
            }
            op = body.call(thisArg, _);
        } catch (e) { op = [6, e]; y = 0; } finally { f = t = 0; }
        if (op[0] & 5) throw op[1]; return { value: op[0] ? op[1] : void 0, done: true };
    }
}

const DEFAULT_VARIANTS = {
  python: {
    template: 'python3 -c "{{src}}"',
    showModal: true,
    appendOutputContents: true,
    showRunButtonInPreview: true,
  },
  javascript: {
    template: 'node -e "{{src}}"',
    showModal: true,
    appendOutputContents: true,
    showRunButtonInPreview: true,
  },
  sh: {
    template: "{{src}}",
    showModal: true,
    appendOutputContents: true,
    showRunButtonInPreview: true,
  },
};

var consts = {
  variants: DEFAULT_VARIANTS,
};

function extract(src, lineNumber, variants = consts.variants) {

    function is(line, target) {
        let str = line.trim();
        return str.toUpperCase() === target.toUpperCase();
    }

    let lines = src.split('\n');
    let begin = null;
    let end = null;
    let lang = null;

    function fenceOpeningWithKey(line) {
        for (var key of Object.keys(variants)) {
            if (is(line, '```' + key)) {
                return key
            }
        }
        return null
    }


    for (let i = lineNumber; i >= 0; i--) {

        let key = fenceOpeningWithKey(lines[i]);
        if (key) {
            begin = i;
            lang = key;
            break
        } else if (i !== lineNumber && is(lines[i], '```')) {
            begin = null;
            lang = null;
            break
        }
    }

    for (let i = lineNumber; i < lines.length; i++) {
        if (i !== begin && is(lines[i], '```')) {
            end = i;
            break
        }
    }

    if ((begin != null) && (end != null)) {
        return {
            lang: lang,
            text: lines.slice(begin + 1, end).join('\n'),
            begin: begin,
            end: end,
        };
    }
    return null

}

var extract_1 = extract;

var path = require('path');
var DEFAULT_SETTINGS = {
    variants: consts.variants
};
var RunSnippets = /** @class */ (function (_super) {
    __extends(RunSnippets, _super);
    function RunSnippets(app, pluginManifest) {
        return _super.call(this, app, pluginManifest) || this;
    }
    RunSnippets.prototype.loadSettings = function () {
        return __awaiter(this, void 0, void 0, function () {
            var _a, _b, _c, _d;
            return __generator(this, function (_e) {
                switch (_e.label) {
                    case 0:
                        _a = this;
                        _c = (_b = Object).assign;
                        _d = [DEFAULT_SETTINGS];
                        return [4 /*yield*/, this.loadData()];
                    case 1:
                        _a.settings = _c.apply(_b, _d.concat([_e.sent()]));
                        return [2 /*return*/];
                }
            });
        });
    };
    RunSnippets.prototype.saveSettings = function () {
        return __awaiter(this, void 0, void 0, function () {
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0: return [4 /*yield*/, this.saveData(this.settings)];
                    case 1:
                        _a.sent();
                        return [2 /*return*/];
                }
            });
        });
    };
    RunSnippets.prototype.onload = function () {
        return __awaiter(this, void 0, void 0, function () {
            var _this = this;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0:
                        console.log("Loading Snippets-plugin");
                        return [4 /*yield*/, this.loadSettings()];
                    case 1:
                        _a.sent();
                        this.addSettingTab(new RunSnippetsSettingsTab(this.app, this));
                        this.addCommand({
                            id: "snippets-plugin",
                            name: "Run",
                            callback: function () { return _this.runSnippet(); },
                            hotkeys: [
                                {
                                    modifiers: ["Mod", "Shift"],
                                    key: "Enter",
                                },
                            ],
                        });
                        this.registerInterval(window.setInterval(this.injectButtons.bind(this), 1000));
                        return [2 /*return*/];
                }
            });
        });
    };
    RunSnippets.prototype.injectButtons = function () {
        this.addRunButtons();
    };
    RunSnippets.prototype.get_vars = function () {
        var active_view = this.app.workspace.getActiveViewOfType(obsidian.MarkdownView);
        if (active_view == null) {
            return null;
        }
        var vaultPath = this.app.vault.adapter.basePath;
        var folder = active_view.file.parent.path;
        var fileName = active_view.file.name;
        return {
            vault_path: vaultPath,
            folder: folder,
            file_name: fileName,
            file_path: path.join(vaultPath, folder, fileName),
            python: 'python3 -c'
        };
    };
    /**
     * Adds buttons for the preview mode
     */
    RunSnippets.prototype.addRunButtons = function () {
        var vars = this.get_vars();
        if (!vars)
            return;
        var variants = this.settings.variants;
        document.querySelectorAll("pre > code").forEach(function (codeBlock) {
            var pre = codeBlock.parentNode;
            var hasButton = pre.parentNode.classList.contains("has-run-button");
            // Already has a button
            if (hasButton) {
                return;
            }
            function definedVariant(classList, variants) {
                for (var _i = 0, _a = Object.keys(variants); _i < _a.length; _i++) {
                    var key = _a[_i];
                    if (classList.contains("language-" + key)) {
                        return key;
                    }
                }
                return null;
            }
            var lang = definedVariant(pre.classList, variants);
            // No variant defined for this language
            if (lang == null) {
                return;
            }
            // @ts-ignore
            var variant = variants[lang];
            // Not active in preview
            if (!variant.showRunButtonInPreview) {
                return;
            }
            pre.parentNode.classList.add("has-run-button");
            var button = document.createElement("button");
            button.className = "run-code-button";
            button.type = "button";
            button.innerText = "Run";
            var src = codeBlock.innerText;
            var command = apply_template(src, variant.template, vars);
            function runCommand(command) {
                var exec = require("child_process").exec;
                button.innerText = "Running";
                exec(command, variant.options ? variant.options : {}, function (error, stdout, stderr) {
                    if (error) {
                        console.error("error: " + error.message);
                        if (variant.showModal) {
                            new obsidian.Notice(error.message);
                        }
                        button.innerText = "error";
                        return;
                    }
                    if (stderr) {
                        console.error("stderr: " + stderr);
                        if (variant.showModal) {
                            new obsidian.Notice(stderr);
                        }
                        button.innerText = "error";
                        return;
                    }
                    console.debug("stdout: " + stdout);
                    if (variant.showModal) {
                        new obsidian.Notice(stdout);
                    }
                    button.innerText = "Run";
                });
            }
            button.addEventListener("click", function () {
                runCommand(command);
            });
            pre.appendChild(button);
        });
    };
    /**
     * rus a snippet, when the cursor is on top of it
     */
    RunSnippets.prototype.runSnippet = function () {
        var vars = this.get_vars();
        if (!vars)
            return;
        var variants = this.settings.variants;
        var view = this.app.workspace.activeLeaf.view;
        if (view instanceof obsidian.MarkdownView) {
            var editor_1 = view.sourceMode.cmEditor;
            var document_1 = editor_1.getDoc().getValue();
            var line = editor_1.getCursor().line;
            var match = extract_1(document_1, line, variants);
            if (match !== null) {
                var targetLine_1 = match.end + 1;
                var lang = match.lang;
                // @ts-ignore
                var variant_1 = variants[lang];
                var command = apply_template(match.text, variant_1.template, vars);
                var exec = require("child_process").exec;
                exec(command, variant_1.options ? variant_1.options : {}, function (error, stdout, stderr) {
                    if (error) {
                        console.error("error: " + error.message);
                        if (variant_1.appendOutputContents) {
                            writeResult(editor_1, error, targetLine_1);
                        }
                        if (variant_1.showModal) {
                            new obsidian.Notice(error.message);
                        }
                        return;
                    }
                    if (stderr) {
                        console.error("stderr: " + stderr);
                        if (variant_1.appendOutputContents) {
                            writeResult(editor_1, stderr, targetLine_1);
                        }
                        if (variant_1.showModal) {
                            new obsidian.Notice(stderr);
                        }
                        return;
                    }
                    console.debug("stdout: " + stdout);
                    if (variant_1.appendOutputContents) {
                        writeResult(editor_1, stdout, targetLine_1);
                    }
                    if (variant_1.showModal) {
                        new obsidian.Notice(stdout);
                    }
                });
            }
        }
    };
    return RunSnippets;
}(obsidian.Plugin));
function writeResult(editor, result, outputLine) {
    if (typeof result === 'string') {
        var output = "\n```output\n" + (result ? result.trim() : result) + "    \n```\n";
        editor.getDoc().replaceRange(output, { line: outputLine, ch: 0 });
    }
}
function apply_template(src, template, vars) {
    var result = template.replace('{{src}}', src);
    result = result.replace('{{vault_path}}', vars.vault_path);
    result = result.replace('{{folder}}', vars.folder);
    result = result.replace('{{file_name}}', vars.file_name);
    result = result.replace('{{file_path}}', vars.file_path);
    return result;
}
var RunSnippetsSettingsTab = /** @class */ (function (_super) {
    __extends(RunSnippetsSettingsTab, _super);
    function RunSnippetsSettingsTab(app, plugin) {
        var _this = _super.call(this, app, plugin) || this;
        _this.plugin = plugin;
        return _this;
    }
    RunSnippetsSettingsTab.prototype.display = function () {
        var _this = this;
        var containerEl = this.containerEl;
        var settings = this.plugin.settings;
        containerEl.empty();
        this.containerEl.createEl("h3", {
            text: "Snippets",
        });
        new obsidian.Setting(containerEl)
            .setName('Code fences')
            .setDesc('config for each language')
            .addTextArea(function (text) {
            text
                .setPlaceholder(JSON.stringify(consts.variants, null, 2))
                .setValue(JSON.stringify(_this.plugin.settings.variants, null, 2) || '')
                .onChange(function (value) { return __awaiter(_this, void 0, void 0, function () {
                var newValue, e_1;
                return __generator(this, function (_a) {
                    switch (_a.label) {
                        case 0:
                            _a.trys.push([0, 2, , 3]);
                            newValue = JSON.parse(value);
                            this.plugin.settings.variants = newValue;
                            return [4 /*yield*/, this.plugin.saveSettings()];
                        case 1:
                            _a.sent();
                            return [3 /*break*/, 3];
                        case 2:
                            e_1 = _a.sent();
                            return [2 /*return*/, false];
                        case 3: return [2 /*return*/];
                    }
                });
            }); });
            text.inputEl.rows = 32;
            text.inputEl.cols = 60;
        });
        this.containerEl.createEl("h4", {
            text: "This plugin is experimental",
        });
    };
    return RunSnippetsSettingsTab;
}(obsidian.PluginSettingTab));

module.exports = RunSnippets;


/* nosourcemap */