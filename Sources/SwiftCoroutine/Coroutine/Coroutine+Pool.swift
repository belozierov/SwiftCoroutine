//
//  Coroutine+Pool.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 19.12.2019.
//  Copyright © 2019 Alex Belozierov. All rights reserved.
//

extension Coroutine {
    
    private static let pool = Pool(maxElements: 32, creator: CoroutineContext.init)
    
    public static func fromPool(with dispatcher: @escaping Dispatcher) -> Coroutine {
        let context = pool.pop()
        let coroutine = Coroutine(context: context, dispatcher: dispatcher)
        coroutine.addHandler { if $0 { pool.push(context) } }
        return coroutine
    }
    
}
