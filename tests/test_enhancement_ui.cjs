const fs=require('fs'),vm=require('vm'),assert=require('assert');
const nodes={};
function node(){return {hidden:false,disabled:false,textContent:'',children:[],append(...args){this.children.push(...args)},replaceChildren(){this.children=[]}};}
let response, posts=0;
const context={console, clearTimeout(){},setTimeout(){return 1},el:()=>node(),$:s=>nodes[s]??=(node()),api:async(url,opts)=>{if(opts)posts++;return response}};
vm.createContext(context);
const source=fs.readFileSync('app/web/app.js','utf8');
vm.runInContext(source.slice(source.indexOf('let enhancementTimer'),source.indexOf('let source = null')),context);
(async()=>{
 response={eligible:true,available:true,job:null};
 await vm.runInContext('enhancementParent="original";refreshEnhancement("original")',context);
 assert(!nodes['#enhancement'].hidden);assert(!nodes['#enhance-button'].disabled);
 await nodes['#enhance-button'].onclick();assert.equal(posts,1);
 response={eligible:true,available:true,job:{id:'child',status:'running',message:'Section 2 of 4'}};
 await vm.runInContext('refreshEnhancement("original")',context);
 assert(nodes['#enhance-button'].hidden);assert.equal(nodes['#enhance-status'].textContent,'Section 2 of 4');
 response={eligible:true,available:false,job:{id:'child',status:'done',video:'/api/jobs/child/video'}};
 await vm.runInContext('refreshEnhancement("original")',context);
 assert.equal(nodes['#enhance-links'].children.length,2);
 assert.equal(nodes['#enhance-links'].children[1].href,'/api/jobs/child/video?download=1');
 response={eligible:true,available:false,job:{id:'child',status:'cancelled'}};
 await vm.runInContext('refreshEnhancement("original")',context);
 assert(nodes['#enhance-button'].disabled);assert(nodes['#enhance-status'].textContent.includes('original is unchanged'));
 vm.runInContext('stopEnhancement()',context);assert(nodes['#enhancement'].hidden);
 console.log('Enhancement UI states passed: available, submit, running, complete, cancelled/offline, navigation.');
})().catch(e=>{console.error(e);process.exitCode=1});
