"""Extract the source features used by the five reported rules; never execute a candidate."""
import ast,contextlib,hashlib,importlib,io,re,shutil,subprocess
import javalang,lizard
from tree_sitter import Language,Parser

CALL={'cpp':'call_expression','java':'method_invocation','python':'call'}
IO={'cpp':set('scanf printf getchar getchar_unlocked putchar putchar_unlocked fread fwrite fgets fputs puts getline'.split()),
    'java':set('nextInt nextLong nextDouble nextFloat nextLine read readLine readInt readLong readDouble write print println printf flush'.split()),
    'python':set('input print read readline readlines write writelines'.split())}
CPP_LOOPS={'for_statement','for_range_loop','while_statement','do_statement'}
def syntax_counts(lang,src,root):
    aliases={};types={};text=lambda n:src[n.start_byte:n.end_byte].decode(errors='replace') if n else ''
    pending=[root]
    while pending:
        n=pending.pop()
        if lang=='python' and n.type=='assignment':
            a=n.child_by_field_name('left');b=n.child_by_field_name('right')
            if a and a.type=='identifier' and b and b.type in ('attribute','identifier'):
                target=re.sub(r'\s+','',text(b))
                if target.split('.')[-1] in IO[lang]:aliases[text(a)]=target
        if lang=='java' and n.type=='variable_declarator':
            name=n.child_by_field_name('name');typ=n.parent.child_by_field_name('type') if n.parent else None
            if name and typ:types[text(name)]=text(typ)
        pending.extend(n.named_children)
    count=sites=0;pending=[root]
    while pending:
        n=pending.pop()
        if n.type in ('comment','line_comment','block_comment'):continue
        count+=int(n.is_named and n.type!='ERROR')
        if n.type==CALL[lang]:
            t=re.sub(r'\s+','',text(n.child_by_field_name('name' if lang=='java' else 'function')))
            t=aliases.get(t,t).split('.')[-1].split('::')[-1];scanner=types.get(text(n.child_by_field_name('object')),'').split('.')[-1]=='Scanner'
            sites+=int(t in IO[lang] or (lang=='java' and scanner and t in ('next','hasNext','hasNextInt','hasNextLong')))
        if lang=='cpp' and n.type=='binary_expression' and (not n.parent or n.parent.type!='binary_expression'):
            stack=[n];stream=False
            while stack:
                part=stack.pop()
                if part.type=='identifier' and text(part) in ('cin','cout','cerr','clog'):stream=True
                stack.extend(part.named_children)
            sites+=int(stream and ('<<' in text(n) or '>>' in text(n)))
        pending.extend(reversed(n.children))
    return dict(ast_nodes=count,io_sites=sites)

class PythonLoops(ast.NodeVisitor):
    def __init__(self):self.total=self.maximum=self.current=self.branches=0
    def visit(self,n):
        self.branches+=isinstance(n,(ast.If,ast.IfExp))
        return super().visit(n)
    def visit_FunctionDef(self,n):
        old=self.current;self.current=0;self.generic_visit(n);self.current=old
    visit_AsyncFunctionDef=visit_FunctionDef
    visit_Lambda=visit_FunctionDef
    def begin(self):
        self.total+=1;self.current+=1;self.maximum=max(self.maximum,self.current)
    def visit_For(self,n):
        self.visit(n.target);self.visit(n.iter);self.body(n)
    visit_AsyncFor=visit_For
    def visit_While(self,n):self.visit(n.test);self.body(n)
    def body(self,n):
        self.begin()
        for z in n.body:self.visit(z)
        self.current-=1
        for z in n.orelse:self.visit(z)
    def visit_ListComp(self,n):
        old=self.current
        for g in n.generators:
            self.visit(g.iter);self.begin();self.visit(g.target);self.branches+=len(g.ifs)
            for z in g.ifs:self.visit(z)
        for z in ([n.elt] if hasattr(n,'elt') else [n.key,n.value]):self.visit(z)
        self.current=old
    visit_SetComp=visit_DictComp=visit_GeneratorExp=visit_ListComp

class Extractor:
    def __init__(self):
        self.parsers={l:Parser(Language(importlib.import_module('tree_sitter_'+l).language())) for l in CALL}
        self.liz=lizard.FileAnalyzer(lizard.get_extensions(['outside']))
    def lizard(self,l,code):
        if code is None:return None,None
        try:
            stream=io.StringIO()
            with contextlib.redirect_stderr(stream):
                metrics=self.liz.analyze_source_code('candidate.'+{'cpp':'cpp','python':'py','java':'java'}[l],code)
            if stream.getvalue():return None,None
            return metrics.nloc,max((x.cyclomatic_complexity for x in metrics.function_list),default=0)
        except Exception:return None,None
    def expanded(self,src,root):
        data=bytearray(src);pending=[root]
        while pending:
            n=pending.pop()
            other=n.type=='preproc_call' and any(c.type=='preproc_directive' and src[c.start_byte:c.end_byte].strip() in (b'#import',b'#include_next') for c in n.named_children)
            if n.type=='preproc_include' or other:
                for i in range(n.start_byte,n.end_byte):
                    if data[i] not in (10,13):data[i]=32
            pending.extend(n.named_children)
        gcc=shutil.which('g++')
        if not gcc:raise RuntimeError('g++ is required for C++')
        try:
            proc=subprocess.run([gcc,'-E','-P','-x','c++','-std=gnu++17','-nostdinc','-nostdinc++','-'],
                input=data.decode(),capture_output=True,encoding='utf8',errors='replace',timeout=8)
            return proc.stdout if proc.returncode==0 and len(proc.stdout)<=4000000 else None
        except (OSError,subprocess.TimeoutExpired):return None
    def structures(self,l,code):
        if l=='python':
            t=PythonLoops();t.visit(ast.parse(code));return t.maximum,t.total,t.branches
        nested=loops=ifs=0
        if l=='cpp':
            root=self.parsers[l].parse(code.encode()).root_node
            if root.has_error:return None,None,None
            queue=[(root,0)]
            while queue:
                n,d=queue.pop()
                if n.type=='comment':continue
                if n.type in ('lambda_expression','function_definition'):d=0
                if n.type in CPP_LOOPS:loops+=1;d+=1;nested=max(nested,d)
                ifs+=n.type in ('if_statement','conditional_expression')
                queue.extend((x,d) for x in reversed(n.named_children))
        else:
            jt=javalang.tree;queue=[(javalang.parse.parse(code),0)]
            while queue:
                n,d=queue.pop()
                if isinstance(n,(list,tuple)):queue.extend((v,d) for v in n);continue
                if not isinstance(n,javalang.ast.Node):continue
                if isinstance(n,(jt.MethodDeclaration,jt.ConstructorDeclaration,jt.LambdaExpression)):d=0
                if isinstance(n,(jt.ForStatement,jt.WhileStatement,jt.DoStatement)):loops+=1;d+=1;nested=max(nested,d)
                ifs+=isinstance(n,(jt.IfStatement,jt.TernaryExpression));queue.extend((v,d) for v in n.children)
        return nested,loops,ifs
    def extract(self,l,code):
        source=code.encode();tree=self.parsers[l].parse(source).root_node
        simple=syntax_counts(l,source,tree);loc,cc=self.lizard(l,code)
        work=code
        if l=='cpp':work=self.expanded(source,tree);_,cc=self.lizard(l,work)
        try:depth,loops,ifs=self.structures(l,work) if work is not None else (None,None,None)
        except Exception:depth=loops=ifs=None
        return {'simple':simple,'tool':dict(raw_nloc=loc,analysis_ccn_max=cc,num_nested_loop=depth,num_loop=loops,num_if=ifs)}
