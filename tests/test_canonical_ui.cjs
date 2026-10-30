const fs=require('fs'),vm=require('vm'),assert=require('assert');
const nodes={};
function node(){return {hidden:false,textContent:'',children:[],append(...args){this.children.push(...args)},replaceChildren(){this.children=[]}};}
let response;
const context={console,Date,location:{hash:'#job=original'},el:()=>node(),$:s=>nodes[s]??=node(),api:async()=>response};
vm.createContext(context);
const source=fs.readFileSync('app/web/app.js','utf8');
vm.runInContext(source.slice(source.indexOf('let canonicalEvidence'),source.indexOf('let enhancementTimer')),context);
(async()=>{
 response={checkpoint:null};
 await vm.runInContext('showCanonicalCheck({id:"original",workflow:"photographic-grounded-v1",status:"running"})',context);
 assert(!nodes['#canonical-check'].hidden);
 response={checkpoint:{title:'Check outpaint 2 of 3',evidence_sha256:'abc',assets:[{name:'raw-outpaint.png',url:'/protected/image'}]},decision:{continue:false,reason:'Geometry is unclear'}};
 await vm.runInContext('showCanonicalCheck({id:"original",workflow:"photographic-grounded-v1",status:"held"})',context);
 assert(nodes['#canonical-check-status'].textContent.includes('held. Geometry is unclear'));
 assert.equal(nodes['#canonical-check-assets'].children[0].src,'/protected/image');
 response.decision={continue:true,reason:'Continuity passed'};
 await vm.runInContext('showCanonicalCheck({id:"original",workflow:"photographic-grounded-v1",status:"done"})',context);
 assert(nodes['#canonical-check-status'].textContent.includes('passed. Continuity passed'));
 assert.equal(nodes['#canonical-check-assets'].children.length,1);
 await vm.runInContext('showCanonicalCheck({id:"legacy",workflow:"photographic-v1",status:"done"})',context);
 assert(nodes['#canonical-check'].hidden);
 console.log('Canonical UI states passed: preparation, hold, terminal update, protected evidence and legacy navigation.');
})().catch(e=>{console.error(e);process.exitCode=1});
