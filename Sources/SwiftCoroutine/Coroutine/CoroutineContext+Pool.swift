//
//  CoroutineContext+Pool.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 19.12.2019.
//  Copyright © 2019 Alex Belozierov. All rights reserved.
//

extension CoroutineContext {
    
    fileprivate static let pool = Pool(maxElements: 32) {
        CoroutineContext(stackSize: Coroutine.StackSize.recommended.size)
    }
    
}

extension Coroutine {
    
    public static func newFromPool(with dispatcher: Dispatcher) -> Coroutine {
        let context = CoroutineContext.pool.pop()
        let coroutine = Coroutine(context: context, dispatcher: dispatcher)
        coroutine.addHandler { if $0 { CoroutineContext.pool.push(context) } }
        return coroutine
    }
    
}

public func coSubroutine(_ block: () -> Void) {
    let context = CoroutineContext.pool.pop()
    defer { CoroutineContext.pool.push(context) }
    CoSubroutine(context: context).start(block)
}
