//
//  CoFuture+Handlers.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 19.01.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

extension CoFuture {
    
    @inlinable @discardableResult
    public func addCancelHandler(on dispatcher: Dispatcher = .sync, execute handler: @escaping () -> Void) -> HandlerIdentifier {
        addErrorHandler(on: dispatcher) { error in
            if let error = error as? CoFutureError,
                error == .cancelled { handler() }
        }
    }
    
}
