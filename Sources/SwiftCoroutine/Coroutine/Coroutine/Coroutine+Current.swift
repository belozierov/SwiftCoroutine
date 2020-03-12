//
//  Coroutine+Current.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 23.12.2019.
//  Copyright © 2019 Alex Belozierov. All rights reserved.
//

extension Coroutine {
    
    @inlinable public static func current() throws -> Coroutine {
        if let coroutine = ThreadCoroutineWrapper.current.coroutine {
            return Coroutine(coroutine: coroutine)
        }
        throw CoroutineError.mustBeCalledInsideCoroutine
    }
    
    @inlinable public static var isInsideCoroutine: Bool {
        ThreadCoroutineWrapper.current.coroutine != nil
    }
    
    @inlinable public var isCurrent: Bool {
        ThreadCoroutineWrapper.current.coroutine === coroutine
    }
    
}
